//! Cost-model operator placement across CPU and GPU devices.

use std::collections::HashMap;

use as_core::{DType, Device, Result};
use as_graph::{Graph, NodeId, Op, TensorMeta};

use crate::registry::{DeviceProfile, DeviceRegistry};

/// Policy controlling how graph operators are placed across available devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    /// Force all operations onto [`Device::Cpu`].
    CpuOnly,
    /// Place all supported compute-heavy operations onto the best available GPU when present.
    PreferGpu,
    /// Roofline cost model: offload to GPU only when `estimated_flops >= min_gpu_flops`
    /// and GPU total cost (compute + launch + transfer) beats CPU cost.
    CostModelAuto {
        /// Minimum FLOPs threshold to consider GPU offload.
        min_gpu_flops: u64,
    },
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self::CostModelAuto {
            min_gpu_flops: 262_144,
        }
    }
}

/// Device assignment and cost breakdown for a single graph node.
#[derive(Debug, Clone, PartialEq)]
pub struct OpPlacement {
    /// Graph node ID.
    pub node_id: NodeId,
    /// Assigned execution device.
    pub device: Device,
    /// Estimated FLOPs for this node.
    pub estimated_flops: u64,
    /// Estimated bytes transferred if inputs live on a different device.
    pub transfer_bytes: usize,
    /// Estimated execution + transfer time in microseconds.
    pub estimated_cost_us: f64,
}

/// Full graph placement plan across devices.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    /// Per-node placement in topological order.
    pub placements: Vec<OpPlacement>,
    /// Quick lookup from [`NodeId`] -> assigned [`Device`].
    pub device_map: HashMap<NodeId, Device>,
    /// Number of nodes assigned to CPU.
    pub cpu_nodes: usize,
    /// Number of nodes assigned to GPU/accelerators.
    pub gpu_nodes: usize,
    /// Number of cross-device tensor transfers required.
    pub cross_device_transfers: usize,
    /// Sum of estimated execution + transfer cost in microseconds.
    pub estimated_total_us: f64,
}

/// Computes a [`PlacementPlan`] for `graph` given its static `metadata`, `registry`, and `policy`.
pub fn plan_placement(
    graph: &Graph,
    metadata: &[TensorMeta],
    registry: &DeviceRegistry,
    policy: PlacementPolicy,
) -> Result<PlacementPlan> {
    let topo = graph.topological_sort()?;
    let cpu_profile = registry.get(Device::Cpu);
    let gpu_profile = registry.best_accelerator();

    let mut placements = Vec::with_capacity(topo.len());
    let mut device_map: HashMap<NodeId, Device> = HashMap::with_capacity(topo.len());
    let mut cpu_nodes = 0usize;
    let mut gpu_nodes = 0usize;
    let mut cross_device_transfers = 0usize;
    let mut estimated_total_us = 0.0f64;

    for node_id in topo {
        let node = graph.node(node_id)?;
        let out_meta = metadata.get(node_id.0);
        let out_bytes = out_meta.map_or(0, |m| m.byte_size().unwrap_or(0));
        let out_dtype = out_meta.map_or(DType::F32, |m| m.dtype);
        let out_numel = out_meta.map_or(0, |m| m.shape.numel());

        let flops = estimate_node_flops(&node.op, &node.inputs, metadata, out_numel);
        let gpu_eligible = is_gpu_eligible(&node.op, out_dtype);

        let chosen_device = match (policy, gpu_profile, gpu_eligible) {
            (PlacementPolicy::CpuOnly, _, _) | (_, None, _) | (_, _, false) => Device::Cpu,
            (PlacementPolicy::PreferGpu, Some(gpu), true) => gpu.device,
            (PlacementPolicy::CostModelAuto { min_gpu_flops }, Some(gpu), true) => {
                if flops < min_gpu_flops {
                    Device::Cpu
                } else {
                    let cpu_cost = estimate_cost_us(
                        cpu_profile,
                        flops,
                        out_bytes,
                        transfer_bytes_for(&node.inputs, metadata, &device_map, Device::Cpu),
                    );
                    let gpu_cost = estimate_cost_us(
                        Some(gpu),
                        flops,
                        out_bytes,
                        transfer_bytes_for(&node.inputs, metadata, &device_map, gpu.device),
                    );
                    if gpu_cost < cpu_cost {
                        gpu.device
                    } else {
                        Device::Cpu
                    }
                }
            }
        };

        let transfer_bytes =
            transfer_bytes_for(&node.inputs, metadata, &device_map, chosen_device);
        if transfer_bytes > 0 {
            cross_device_transfers += 1;
        }

        let target_profile = registry.get(chosen_device).or(cpu_profile);
        let cost_us = estimate_cost_us(target_profile, flops, out_bytes, transfer_bytes);

        if chosen_device == Device::Cpu {
            cpu_nodes += 1;
        } else {
            gpu_nodes += 1;
        }
        estimated_total_us += cost_us;
        device_map.insert(node_id, chosen_device);
        placements.push(OpPlacement {
            node_id,
            device: chosen_device,
            estimated_flops: flops,
            transfer_bytes,
            estimated_cost_us: cost_us,
        });
    }

    Ok(PlacementPlan {
        placements,
        device_map,
        cpu_nodes,
        gpu_nodes,
        cross_device_transfers,
        estimated_total_us,
    })
}

fn is_gpu_eligible(op: &Op, dtype: DType) -> bool {
    dtype == DType::F32
        && matches!(
            op,
            Op::MatMul | Op::FusedLinear { .. } | Op::Binary(as_backend::BinaryOp::Add)
        )
}

fn transfer_bytes_for(
    inputs: &[NodeId],
    metadata: &[TensorMeta],
    device_map: &HashMap<NodeId, Device>,
    target: Device,
) -> usize {
    let mut total = 0usize;
    for &inp in inputs {
        let src_dev = device_map.get(&inp).copied().unwrap_or(Device::Cpu);
        if src_dev != target
            && let Some(meta) = metadata.get(inp.0)
        {
            total = total.saturating_add(meta.byte_size().unwrap_or(0));
        }
    }
    total
}

fn estimate_cost_us(
    profile: Option<&DeviceProfile>,
    flops: u64,
    out_bytes: usize,
    transfer_bytes: usize,
) -> f64 {
    let Some(p) = profile else {
        return 0.0;
    };
    let compute_us = (flops as f64) / (p.peak_gflops_estimate.max(1.0) * 1e3);
    let mem_us = (out_bytes as f64) / (p.memory_bandwidth_gbs.max(1.0) * 1e3);
    let xfer_us = if transfer_bytes == 0 || !p.transfer_bandwidth_gbs.is_finite() {
        0.0
    } else {
        (transfer_bytes as f64) / (p.transfer_bandwidth_gbs.max(1.0) * 1e3)
    };
    p.launch_overhead_us + compute_us.max(mem_us) + xfer_us
}

fn estimate_node_flops(
    op: &Op,
    inputs: &[NodeId],
    metadata: &[TensorMeta],
    out_numel: usize,
) -> u64 {
    match op {
        Op::Input { .. } | Op::Constant { .. } | Op::Reshape(_) | Op::Transpose { .. } => 0,
        Op::Binary(_) | Op::Unary(_) => out_numel as u64,
        Op::FusedUnaryChain(c) => (out_numel as u64).saturating_mul(c.len() as u64),
        Op::Reduce { .. } => inputs
            .first()
            .and_then(|id| metadata.get(id.0))
            .map_or(out_numel as u64, |m| m.shape.numel() as u64),
        Op::MatMul | Op::FusedLinear { .. } => {
            let k = inputs
                .first()
                .and_then(|id| metadata.get(id.0))
                .and_then(|m| m.shape.dims().last().copied())
                .unwrap_or(0) as u64;
            (out_numel as u64).saturating_mul(k).saturating_mul(2)
        }
    }
}
