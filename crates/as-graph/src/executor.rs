//! Memory-scheduled forward execution engine for [`Graph`].

use std::collections::HashMap;

use as_backend::{BinaryOp, ReduceOp, UnaryOp};
use as_core::{DType, Error, Result};
use as_tensor::Tensor;

use crate::ir::{Graph, NodeId, Op};

/// Compiled execution schedule with liveness metadata for releasing dead buffers.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Deterministic topological execution order.
    pub order: Vec<NodeId>,
    /// Remaining consumer reference counts per node (outputs add +1 so they are retained).
    pub ref_counts: Vec<usize>,
}

impl ExecutionPlan {
    /// Compiles an [`ExecutionPlan`] from a [`Graph`].
    pub fn compile(graph: &Graph) -> Result<Self> {
        let order = graph.topological_sort()?;
        let n = graph.nodes().len();
        let mut ref_counts = vec![0usize; n];
        for node in graph.nodes() {
            for &inp in &node.inputs {
                if let Some(rc) = ref_counts.get_mut(inp.0) {
                    *rc += 1;
                }
            }
        }
        for &out in graph.outputs() {
            if let Some(rc) = ref_counts.get_mut(out.0) {
                *rc += 1;
            }
        }
        Ok(Self { order, ref_counts })
    }
}

/// Executes `graph` on the CPU backend using an [`ExecutionPlan`], freeing dead
/// intermediate tensors immediately after their last consumer finishes.
pub fn execute_graph(
    graph: &Graph,
    inputs: &HashMap<NodeId, Tensor>,
) -> Result<HashMap<NodeId, Tensor>> {
    let plan = ExecutionPlan::compile(graph)?;
    let mut remaining_refs = plan.ref_counts;
    let mut live_values: HashMap<NodeId, Tensor> = HashMap::new();

    for node_id in plan.order {
        let node = graph.node(node_id)?;
        let out_tensor = match &node.op {
            Op::Input {
                name,
                shape,
                dtype,
            } => {
                let tensor = inputs.get(&node_id).ok_or_else(|| Error::InvalidArgument {
                    context: "execute_graph",
                    reason: format!("missing required input `{name}` (node {})", node_id.0),
                })?;
                if tensor.shape() != shape {
                    return Err(Error::ShapeMismatch {
                        context: "execute_graph input",
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
                tensor.clone()
            }
            Op::Constant {
                shape,
                dtype,
                values,
            } => {
                let base = Tensor::from_f64(values, shape.dims().to_vec())?;
                if *dtype == DType::F64 {
                    base
                } else {
                    base.to_dtype(*dtype)?
                }
            }
            Op::Binary(bin_op) => {
                let lhs = fetch_input(&live_values, &node.inputs, 0)?;
                let rhs = fetch_input(&live_values, &node.inputs, 1)?;
                match bin_op {
                    BinaryOp::Add => lhs.add(rhs)?,
                    BinaryOp::Sub => lhs.sub(rhs)?,
                    BinaryOp::Mul => lhs.mul(rhs)?,
                    BinaryOp::Div => lhs.div(rhs)?,
                }
            }
            Op::Unary(u_op) => {
                let src = fetch_input(&live_values, &node.inputs, 0)?;
                apply_unary(src, *u_op)?
            }
            Op::FusedUnaryChain(chain) => {
                let src = fetch_input(&live_values, &node.inputs, 0)?;
                apply_fused_unary_chain(src, chain)?
            }
            Op::MatMul => {
                let lhs = fetch_input(&live_values, &node.inputs, 0)?;
                let rhs = fetch_input(&live_values, &node.inputs, 1)?;
                lhs.matmul(rhs)?
            }
            Op::Reshape(target_shape) => {
                let src = fetch_input(&live_values, &node.inputs, 0)?;
                src.reshape(target_shape.dims().to_vec())?
            }
            Op::Transpose { axis_a, axis_b } => {
                let src = fetch_input(&live_values, &node.inputs, 0)?;
                src.transpose(*axis_a, *axis_b)?
            }
            Op::Reduce { op, axis, keepdim } => {
                let src = fetch_input(&live_values, &node.inputs, 0)?;
                match (op, axis) {
                    (ReduceOp::Sum, None) => src.sum()?,
                    (ReduceOp::Sum, Some(ax)) => src.sum_axis(*ax, *keepdim)?,
                    (ReduceOp::Mean, None) => src.mean()?,
                    (ReduceOp::Mean, Some(ax)) => src.mean_axis(*ax, *keepdim)?,
                    (ReduceOp::Max, None) => src.max()?,
                    (ReduceOp::Max, Some(ax)) => src.max_axis(*ax, *keepdim)?,
                    (ReduceOp::Min, None) => src.min()?,
                    (ReduceOp::Min, Some(ax)) => src.min_axis(*ax, *keepdim)?,
                }
            }
        };

        // Release consumed intermediates whose ref count has hit zero
        for &inp_id in &node.inputs {
            if let Some(rc) = remaining_refs.get_mut(inp_id.0) {
                *rc = rc.saturating_sub(1);
                if *rc == 0 {
                    live_values.remove(&inp_id);
                }
            }
        }

        if remaining_refs.get(node_id.0).copied().unwrap_or(0) > 0 {
            live_values.insert(node_id, out_tensor);
        }
    }

    let mut results = HashMap::with_capacity(graph.outputs().len());
    for &out_id in graph.outputs() {
        if let Some(t) = live_values.remove(&out_id) {
            results.insert(out_id, t);
        }
    }
    Ok(results)
}

fn fetch_input<'a>(
    values: &'a HashMap<NodeId, Tensor>,
    inputs: &[NodeId],
    pos: usize,
) -> Result<&'a Tensor> {
    let id = inputs.get(pos).copied().ok_or(Error::InternalInvariant {
        message: "missing node input slot during execution",
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

/// Executes a fused unary chain in a single pass over host `f32`/`f64` buffers,
/// avoiding intermediate tensor allocations.
fn apply_fused_unary_chain(src: &Tensor, chain: &[UnaryOp]) -> Result<Tensor> {
    match src.dtype() {
        DType::F32 => {
            let mut vals = src.to_f32_vec()?;
            for x in &mut vals {
                let mut v = *x;
                for &op in chain {
                    v = eval_unary_f32(op, v);
                }
                *x = v;
            }
            Tensor::from_f32(&vals, src.dims().to_vec())
        }
        DType::F64 => {
            let mut vals = src.to_f64_vec()?;
            for x in &mut vals {
                let mut v = *x;
                for &op in chain {
                    v = eval_unary_f64(op, v);
                }
                *x = v;
            }
            Tensor::from_f64(&vals, src.dims().to_vec())
        }
        _ => {
            let mut cur = src.clone();
            for &op in chain {
                cur = apply_unary(&cur, op)?;
            }
            Ok(cur)
        }
    }
}

#[inline]
fn eval_unary_f32(op: UnaryOp, x: f32) -> f32 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Relu => {
            if x.is_nan() {
                x
            } else {
                x.max(0.0)
            }
        }
        UnaryOp::Powf(p) => x.powf(p as f32),
    }
}

#[inline]
fn eval_unary_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Relu => {
            if x.is_nan() {
                x
            } else {
                x.max(0.0)
            }
        }
        UnaryOp::Powf(p) => x.powf(p),
    }
}
