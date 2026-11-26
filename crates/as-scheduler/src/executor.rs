//! Multi-device, wave-parallel graph execution engine.

use std::collections::HashMap;
use std::time::Instant;

use as_backend::{Backend, BinaryOp, ReduceOp, UnaryOp};
use as_core::{DType, Device, Error, MemoryLimits, Result};
use as_cpu::{CpuBuffer, CpuStorage};
use as_graph::{ExecutionPlan, Graph, NodeId, Op, infer_graph_metadata};
use as_metal::MetalBackend;
use as_tensor::Tensor;

use crate::placement::{PlacementPlan, PlacementPolicy, plan_placement};
use crate::registry::DeviceRegistry;
use crate::wave::{WaveSchedule, build_wave_schedule};

/// Execution telemetry returned by [`HardwareScheduler::execute`].
#[derive(Debug, Clone)]
pub struct ScheduleExecutionReport {
    /// Total execution time in microseconds.
    pub total_duration_us: u128,
    /// Number of sequential DAG waves executed.
    pub waves_executed: usize,
    /// Number of waves that dispatched multiple nodes concurrently.
    pub parallel_waves: usize,
    /// Number of ops executed on CPU.
    pub cpu_ops_executed: usize,
    /// Number of ops offloaded to Metal GPU.
    pub gpu_ops_executed: usize,
    /// Underlying placement plan used.
    pub placement: PlacementPlan,
    /// Underlying DAG wave schedule used.
    pub wave_schedule: WaveSchedule,
}

/// Hardware Scheduler orchestrating CPU multi-threading and Apple Silicon Metal GPU offload.
pub struct HardwareScheduler {
    registry: DeviceRegistry,
    policy: PlacementPolicy,
    metal: Option<MetalBackend>,
    limits: MemoryLimits,
}

impl HardwareScheduler {
    /// Creates a [`HardwareScheduler`] by probing host hardware with `policy`.
    pub fn new(policy: PlacementPolicy) -> Self {
        let registry = DeviceRegistry::probe_host();
        let metal = MetalBackend::new().ok();
        Self {
            registry,
            policy,
            metal,
            limits: MemoryLimits::default(),
        }
    }

    /// Creates a [`HardwareScheduler`] with explicit `registry`, `policy`, and `limits`.
    pub fn with_registry(
        registry: DeviceRegistry,
        policy: PlacementPolicy,
        limits: MemoryLimits,
    ) -> Self {
        let metal = if registry
            .get(Device::Metal(0))
            .is_some_and(|p| p.available)
        {
            MetalBackend::with_limits(limits).ok()
        } else {
            None
        };
        Self {
            registry,
            policy,
            metal,
            limits,
        }
    }

    /// Returns the discovered [`DeviceRegistry`].
    pub fn registry(&self) -> &DeviceRegistry {
        &self.registry
    }

    /// Returns the active [`PlacementPolicy`].
    pub const fn policy(&self) -> PlacementPolicy {
        self.policy
    }

    /// Executes `graph` using wave-parallel dispatch and cost-model CPU/GPU placement.
    pub fn execute(
        &self,
        graph: &Graph,
        inputs: &HashMap<NodeId, Tensor>,
    ) -> Result<(HashMap<NodeId, Tensor>, ScheduleExecutionReport)> {
        let t0 = Instant::now();
        let metadata = infer_graph_metadata(graph)?;
        let placement = plan_placement(graph, &metadata, &self.registry, self.policy)?;
        let wave_schedule = build_wave_schedule(graph)?;
        let exec_plan = ExecutionPlan::compile(graph)?;

        let mut remaining_refs = exec_plan.ref_counts;
        let mut live_values: HashMap<NodeId, Tensor> = HashMap::new();
        let mut parallel_waves = 0usize;
        let mut cpu_ops_executed = 0usize;
        let mut gpu_ops_executed = 0usize;

        for wave in &wave_schedule.waves {
            if wave.nodes.len() <= 1 {
                for &node_id in &wave.nodes {
                    let target_dev = placement
                        .device_map
                        .get(&node_id)
                        .copied()
                        .unwrap_or(Device::Cpu);
                    let (tensor, used_gpu) =
                        self.eval_node(graph, node_id, inputs, &live_values, target_dev)?;
                    if used_gpu {
                        gpu_ops_executed += 1;
                    } else {
                        cpu_ops_executed += 1;
                    }
                    commit_node(graph, node_id, tensor, &mut remaining_refs, &mut live_values)?;
                }
            } else {
                parallel_waves += 1;
                let live_ref = &live_values;
                let wave_results: Vec<Result<(NodeId, Tensor, bool)>> =
                    std::thread::scope(|scope| {
                        let handles: Vec<_> = wave
                            .nodes
                            .iter()
                            .map(|&node_id| {
                                let target_dev = placement
                                    .device_map
                                    .get(&node_id)
                                    .copied()
                                    .unwrap_or(Device::Cpu);
                                scope.spawn(move || {
                                    let (t, used_gpu) = self.eval_node(
                                        graph,
                                        node_id,
                                        inputs,
                                        live_ref,
                                        target_dev,
                                    )?;
                                    Ok((node_id, t, used_gpu))
                                })
                            })
                            .collect();

                        handles
                            .into_iter()
                            .map(|h| {
                                h.join().map_err(|_| Error::InternalInvariant {
                                    message: "scheduler wave worker thread panicked",
                                })?
                            })
                            .collect()
                    });

                for res in wave_results {
                    let (node_id, tensor, used_gpu) = res?;
                    if used_gpu {
                        gpu_ops_executed += 1;
                    } else {
                        cpu_ops_executed += 1;
                    }
                    commit_node(graph, node_id, tensor, &mut remaining_refs, &mut live_values)?;
                }
            }
        }

        let mut outputs = HashMap::with_capacity(graph.outputs().len());
        for &out_id in graph.outputs() {
            if let Some(t) = live_values.remove(&out_id) {
                outputs.insert(out_id, t);
            }
        }

        let report = ScheduleExecutionReport {
            total_duration_us: t0.elapsed().as_micros(),
            waves_executed: wave_schedule.waves.len(),
            parallel_waves,
            cpu_ops_executed,
            gpu_ops_executed,
            placement,
            wave_schedule,
        };

        Ok((outputs, report))
    }

    fn eval_node(
        &self,
        graph: &Graph,
        node_id: NodeId,
        inputs: &HashMap<NodeId, Tensor>,
        live_values: &HashMap<NodeId, Tensor>,
        target_dev: Device,
    ) -> Result<(Tensor, bool)> {
        let node = graph.node(node_id)?;
        let use_metal = target_dev == Device::Metal(0) && self.metal.is_some();

        match &node.op {
            Op::Input {
                name,
                shape,
                dtype,
            } => {
                let tensor = inputs.get(&node_id).ok_or_else(|| Error::InvalidArgument {
                    context: "HardwareScheduler::execute",
                    reason: format!("missing required input `{name}` (node {})", node_id.0),
                })?;
                if tensor.shape() != shape {
                    return Err(Error::ShapeMismatch {
                        context: "HardwareScheduler input",
                        expected: shape.dims().to_vec(),
                        actual: tensor.dims().to_vec(),
                    });
                }
                if tensor.dtype() != *dtype {
                    return Err(Error::DTypeMismatch {
                        expected: *dtype,
                        actual: tensor.dtype(),
                    });
                }
                Ok((tensor.clone(), false))
            }
            Op::Constant {
                shape,
                dtype,
                values,
            } => {
                let base = Tensor::from_f64(values, shape.dims().to_vec())?;
                let t = if *dtype == DType::F64 {
                    base
                } else {
                    base.to_dtype(*dtype)?
                };
                Ok((t, false))
            }
            Op::MatMul => {
                let lhs = fetch_input(live_values, &node.inputs, 0)?;
                let rhs = fetch_input(live_values, &node.inputs, 1)?;
                if use_metal
                    && lhs.dtype() == DType::F32
                    && rhs.dtype() == DType::F32
                    && lhs.rank() == 2
                    && rhs.rank() == 2
                    && let Some(metal) = &self.metal
                {
                    let out = metal_matmul_f32(metal, lhs, rhs, self.limits)?;
                    return Ok((out, true));
                }
                Ok((lhs.matmul(rhs)?, false))
            }
            Op::FusedLinear {
                has_bias,
                activation,
            } => {
                let x = fetch_input(live_values, &node.inputs, 0)?;
                let w = fetch_input(live_values, &node.inputs, 1)?;
                let (mut out, used_gpu) = if use_metal
                    && x.dtype() == DType::F32
                    && w.dtype() == DType::F32
                    && x.rank() == 2
                    && w.rank() == 2
                    && let Some(metal) = &self.metal
                {
                    (metal_matmul_f32(metal, x, w, self.limits)?, true)
                } else {
                    (x.matmul(w)?, false)
                };
                if *has_bias {
                    let bias = fetch_input(live_values, &node.inputs, 2)?;
                    out = out.add(bias)?;
                }
                if let Some(act) = activation {
                    out = apply_unary(&out, *act)?;
                }
                Ok((out, used_gpu))
            }
            Op::Binary(bin_op) => {
                let lhs = fetch_input(live_values, &node.inputs, 0)?;
                let rhs = fetch_input(live_values, &node.inputs, 1)?;
                let out = match bin_op {
                    BinaryOp::Add => lhs.add(rhs)?,
                    BinaryOp::Sub => lhs.sub(rhs)?,
                    BinaryOp::Mul => lhs.mul(rhs)?,
                    BinaryOp::Div => lhs.div(rhs)?,
                };
                Ok((out, false))
            }
            Op::Unary(u_op) => {
                let src = fetch_input(live_values, &node.inputs, 0)?;
                Ok((apply_unary(src, *u_op)?, false))
            }
            Op::FusedUnaryChain(chain) => {
                let mut cur = fetch_input(live_values, &node.inputs, 0)?.clone();
                for &op in chain {
                    cur = apply_unary(&cur, op)?;
                }
                Ok((cur, false))
            }
            Op::Reshape(target) => {
                let src = fetch_input(live_values, &node.inputs, 0)?;
                Ok((src.reshape(target.dims().to_vec())?, false))
            }
            Op::Transpose { axis_a, axis_b } => {
                let src = fetch_input(live_values, &node.inputs, 0)?;
                Ok((src.transpose(*axis_a, *axis_b)?, false))
            }
            Op::Reduce { op, axis, keepdim } => {
                let src = fetch_input(live_values, &node.inputs, 0)?;
                let out = match (op, axis) {
                    (ReduceOp::Sum, None) => src.sum()?,
                    (ReduceOp::Sum, Some(ax)) => src.sum_axis(*ax, *keepdim)?,
                    (ReduceOp::Mean, None) => src.mean()?,
                    (ReduceOp::Mean, Some(ax)) => src.mean_axis(*ax, *keepdim)?,
                    (ReduceOp::Max, None) => src.max()?,
                    (ReduceOp::Max, Some(ax)) => src.max_axis(*ax, *keepdim)?,
                    (ReduceOp::Min, None) => src.min()?,
                    (ReduceOp::Min, Some(ax)) => src.min_axis(*ax, *keepdim)?,
                };
                Ok((out, false))
            }
        }
    }
}

fn metal_matmul_f32(
    metal: &MetalBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    _limits: MemoryLimits,
) -> Result<Tensor> {
    let lhs_storage = CpuStorage::from_buffer(CpuBuffer::F32(lhs.to_f32_vec()?));
    let rhs_storage = CpuStorage::from_buffer(CpuBuffer::F32(rhs.to_f32_vec()?));
    let lhs_layout = as_core::Layout::contiguous(lhs.shape().clone());
    let rhs_layout = as_core::Layout::contiguous(rhs.shape().clone());
    let (out_storage, out_shape) =
        metal.matmul(&lhs_storage, &lhs_layout, &rhs_storage, &rhs_layout)?;
    let CpuBuffer::F32(vals) = out_storage.buffer() else {
        return Err(Error::InternalInvariant {
            message: "metal_matmul_f32 expected F32 output",
        });
    };
    Tensor::from_f32(vals, out_shape.dims().to_vec())
}

fn commit_node(
    graph: &Graph,
    node_id: NodeId,
    tensor: Tensor,
    remaining_refs: &mut [usize],
    live_values: &mut HashMap<NodeId, Tensor>,
) -> Result<()> {
    let node = graph.node(node_id)?;
    for &inp_id in &node.inputs {
        if let Some(rc) = remaining_refs.get_mut(inp_id.0) {
            *rc = rc.saturating_sub(1);
            if *rc == 0 {
                live_values.remove(&inp_id);
            }
        }
    }
    if remaining_refs.get(node_id.0).copied().unwrap_or(0) > 0 {
        live_values.insert(node_id, tensor);
    }
    Ok(())
}

fn fetch_input<'a>(
    values: &'a HashMap<NodeId, Tensor>,
    inputs: &[NodeId],
    pos: usize,
) -> Result<&'a Tensor> {
    let id = inputs.get(pos).copied().ok_or(Error::InternalInvariant {
        message: "missing node input slot during scheduled execution",
    })?;
    values.get(&id).ok_or(Error::InternalInvariant {
        message: "scheduled dependency tensor not materialized",
    })
}

fn apply_unary(src: &Tensor, op: UnaryOp) -> Result<Tensor> {
    match op {
        UnaryOp::Neg => src.neg(),
        UnaryOp::Abs => src.abs(),
        UnaryOp::Exp => src.exp(),
        UnaryOp::Ln => src.ln(),
        UnaryOp::Sqrt => src.sqrt(),
        UnaryOp::Relu => src.relu(),
        UnaryOp::Powf(p) => src.powf(p),
    }
}
