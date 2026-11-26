//! Static shape/dtype inference and graph-to-memory-plan lowering.

use as_core::{DType, Error, Result, Shape};
use as_memory::{MemoryPlan, TensorLifetime, plan_memory_reuse};

use crate::executor::ExecutionPlan;
use crate::ir::{Graph, Op};

/// Statically inferred tensor shape and data type for a graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    /// Inferred tensor shape.
    pub shape: Shape,
    /// Inferred element data type.
    pub dtype: DType,
}

impl TensorMeta {
    /// Computes the exact byte size of this tensor, guarding against overflow.
    pub fn byte_size(&self) -> Result<usize> {
        self.shape
            .numel()
            .checked_mul(self.dtype.size_in_bytes())
            .ok_or(Error::ArithmeticOverflow {
                context: "TensorMeta::byte_size",
            })
    }
}

/// Statically infers [`TensorMeta`] (shape & dtype) for every node in `graph`.
pub fn infer_graph_metadata(graph: &Graph) -> Result<Vec<TensorMeta>> {
    let topo = graph.topological_sort()?;
    let n = graph.nodes().len();
    let mut metas: Vec<Option<TensorMeta>> = vec![None; n];

    for node_id in topo {
        let node = graph.node(node_id)?;
        let meta = match &node.op {
            Op::Input { shape, dtype, .. } | Op::Constant { shape, dtype, .. } => TensorMeta {
                shape: shape.clone(),
                dtype: *dtype,
            },
            Op::Unary(_) | Op::FusedUnaryChain(_) => {
                let src = require_meta(&metas, &node.inputs, 0)?;
                src.clone()
            }
            Op::Binary(_) => {
                let lhs = require_meta(&metas, &node.inputs, 0)?;
                let rhs = require_meta(&metas, &node.inputs, 1)?;
                if lhs.dtype != rhs.dtype {
                    return Err(Error::DTypeMismatch {
                        expected: lhs.dtype,
                        actual: rhs.dtype,
                    });
                }
                let shape = lhs.shape.broadcast_shape(&rhs.shape)?;
                TensorMeta {
                    shape,
                    dtype: lhs.dtype,
                }
            }
            Op::MatMul => {
                let lhs = require_meta(&metas, &node.inputs, 0)?;
                let rhs = require_meta(&metas, &node.inputs, 1)?;
                if lhs.dtype != rhs.dtype {
                    return Err(Error::DTypeMismatch {
                        expected: lhs.dtype,
                        actual: rhs.dtype,
                    });
                }
                let shape = infer_matmul_shape(&lhs.shape, &rhs.shape)?;
                TensorMeta {
                    shape,
                    dtype: lhs.dtype,
                }
            }
            Op::FusedLinear { has_bias, .. } => {
                let lhs = require_meta(&metas, &node.inputs, 0)?;
                let rhs = require_meta(&metas, &node.inputs, 1)?;
                if lhs.dtype != rhs.dtype {
                    return Err(Error::DTypeMismatch {
                        expected: lhs.dtype,
                        actual: rhs.dtype,
                    });
                }
                let mut shape = infer_matmul_shape(&lhs.shape, &rhs.shape)?;
                if *has_bias {
                    let bias = require_meta(&metas, &node.inputs, 2)?;
                    if bias.dtype != lhs.dtype {
                        return Err(Error::DTypeMismatch {
                            expected: lhs.dtype,
                            actual: bias.dtype,
                        });
                    }
                    shape = shape.broadcast_shape(&bias.shape)?;
                }
                TensorMeta {
                    shape,
                    dtype: lhs.dtype,
                }
            }
            Op::Reshape(target) => {
                let src = require_meta(&metas, &node.inputs, 0)?;
                if src.shape.numel() != target.numel() {
                    return Err(Error::ShapeMismatch {
                        context: "infer_graph_metadata Reshape",
                        expected: target.dims().to_vec(),
                        actual: src.shape.dims().to_vec(),
                    });
                }
                TensorMeta {
                    shape: target.clone(),
                    dtype: src.dtype,
                }
            }
            Op::Transpose { axis_a, axis_b } => {
                let src = require_meta(&metas, &node.inputs, 0)?;
                let rank = src.shape.rank();
                if *axis_a >= rank || *axis_b >= rank {
                    return Err(Error::InvalidAxis {
                        axis: (*axis_a).max(*axis_b),
                        rank,
                    });
                }
                let mut dims = src.shape.dims().to_vec();
                dims.swap(*axis_a, *axis_b);
                TensorMeta {
                    shape: Shape::new(dims)?,
                    dtype: src.dtype,
                }
            }
            Op::Reduce { axis, keepdim, .. } => {
                let src = require_meta(&metas, &node.inputs, 0)?;
                let shape = match axis {
                    None => {
                        if *keepdim {
                            Shape::new(vec![1; src.shape.rank()])?
                        } else {
                            Shape::new(vec![1])?
                        }
                    }
                    Some(ax) => {
                        let rank = src.shape.rank();
                        if *ax >= rank {
                            return Err(Error::InvalidAxis { axis: *ax, rank });
                        }
                        let mut dims = src.shape.dims().to_vec();
                        if *keepdim {
                            if let Some(d) = dims.get_mut(*ax) {
                                *d = 1;
                            }
                        } else {
                            dims.remove(*ax);
                            if dims.is_empty() {
                                dims.push(1);
                            }
                        }
                        Shape::new(dims)?
                    }
                };
                TensorMeta {
                    shape,
                    dtype: src.dtype,
                }
            }
        };

        if let Some(slot) = metas.get_mut(node_id.0) {
            *slot = Some(meta);
        }
    }

    let mut result = Vec::with_capacity(n);
    for m in metas {
        result.push(m.ok_or(Error::InternalInvariant {
            message: "infer_graph_metadata missed node",
        })?);
    }
    Ok(result)
}

/// Computes a static buffer-reuse [`MemoryPlan`] from a graph, its [`ExecutionPlan`], and inferred [`TensorMeta`].
pub fn build_memory_plan(
    graph: &Graph,
    plan: &ExecutionPlan,
    metadata: &[TensorMeta],
) -> Result<MemoryPlan> {
    let n = graph.nodes().len();
    let mut step_of = vec![0usize; n];
    for (step, &node_id) in plan.order.iter().enumerate() {
        if let Some(s) = step_of.get_mut(node_id.0) {
            *s = step;
        }
    }

    let mut last_use = step_of.clone();
    for (step, &node_id) in plan.order.iter().enumerate() {
        let node = graph.node(node_id)?;
        for &inp in &node.inputs {
            if let Some(lu) = last_use.get_mut(inp.0) {
                *lu = (*lu).max(step);
            }
        }
    }

    let mut is_output = vec![false; n];
    for &out_id in graph.outputs() {
        if let Some(flag) = is_output.get_mut(out_id.0) {
            *flag = true;
        }
    }

    let mut lifetimes = Vec::with_capacity(n);
    for (idx, node) in graph.nodes().iter().enumerate() {
        let meta = metadata.get(idx).ok_or(Error::InternalInvariant {
            message: "metadata length mismatch in build_memory_plan",
        })?;
        let first_step = step_of.get(idx).copied().unwrap_or(0);
        let last_step = last_use.get(idx).copied().unwrap_or(first_step);
        let pinned = matches!(node.op, Op::Input { .. })
            || is_output.get(idx).copied().unwrap_or(false);

        lifetimes.push(TensorLifetime {
            id: idx,
            byte_size: meta.byte_size()?,
            first_step,
            last_step,
            pinned,
        });
    }

    plan_memory_reuse(&lifetimes)
}

fn require_meta<'a>(
    metas: &'a [Option<TensorMeta>],
    inputs: &[crate::ir::NodeId],
    pos: usize,
) -> Result<&'a TensorMeta> {
    let id = inputs.get(pos).copied().ok_or(Error::InternalInvariant {
        message: "missing node input during shape inference",
    })?;
    metas
        .get(id.0)
        .and_then(Option::as_ref)
        .ok_or(Error::InternalInvariant {
            message: "dependency metadata not inferred yet",
        })
}

fn infer_matmul_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let l_rank = lhs.rank();
    let r_rank = rhs.rank();
    if l_rank < 2 || r_rank < 2 {
        return Err(Error::InvalidRank {
            context: "infer_matmul_shape",
            rank: l_rank.min(r_rank),
            expected: 2,
        });
    }
    let l_dims = lhs.dims();
    let r_dims = rhs.dims();
    let m = l_dims.get(l_rank - 2).copied().unwrap_or(0);
    let k1 = l_dims.get(l_rank - 1).copied().unwrap_or(0);
    let k2 = r_dims.get(r_rank - 2).copied().unwrap_or(0);
    let n = r_dims.get(r_rank - 1).copied().unwrap_or(0);
    if k1 != k2 {
        return Err(Error::ShapeMismatch {
            context: "infer_matmul_shape inner dimension",
            expected: vec![k1],
            actual: vec![k2],
        });
    }
    let l_batch = Shape::new(l_dims.get(..l_rank - 2).unwrap_or(&[]).to_vec())?;
    let r_batch = Shape::new(r_dims.get(..r_rank - 2).unwrap_or(&[]).to_vec())?;
    let batch = if l_batch.rank() == 0 && r_batch.rank() == 0 {
        Vec::new()
    } else if l_batch.rank() == 0 {
        r_batch.dims().to_vec()
    } else if r_batch.rank() == 0 {
        l_batch.dims().to_vec()
    } else {
        l_batch.broadcast_shape(&r_batch)?.dims().to_vec()
    };

    let mut out = batch;
    out.push(m);
    out.push(n);
    Shape::new(out)
}
