//! Core Intermediate Representation (IR) for computational graphs.

use std::collections::VecDeque;

use as_backend::{BinaryOp, ReduceOp, UnaryOp};
use as_core::{DType, Error, Result, Shape};

/// Maximum permitted nodes in a single graph to guard against DoS on untrusted graphs.
pub const MAX_GRAPH_NODES: usize = 1_000_000;

/// Strongly typed identifier for a node in a [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Computation graph operator IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// External runtime input placeholder.
    Input {
        /// Human-readable input name.
        name: String,
        /// Expected input shape.
        shape: Shape,
        /// Expected input dtype.
        dtype: DType,
    },
    /// Embedded constant tensor (stored in `f64` canonical form and cast to `dtype` at execution).
    Constant {
        /// Constant shape.
        shape: Shape,
        /// Target dtype.
        dtype: DType,
        /// Row-major values.
        values: Vec<f64>,
    },
    /// Element-wise binary operation with broadcasting.
    Binary(BinaryOp),
    /// Single element-wise unary operation.
    Unary(UnaryOp),
    /// Fused chain of consecutive unary operations (produced by the optimizer).
    FusedUnaryChain(Vec<UnaryOp>),
    /// 2D or batched N-D matrix multiplication.
    MatMul,
    /// Tensor reshape.
    Reshape(Shape),
    /// Zero-copy axis swap.
    Transpose {
        /// First axis.
        axis_a: usize,
        /// Second axis.
        axis_b: usize,
    },
    /// Global or axis-wise reduction.
    Reduce {
        /// Reduction operator.
        op: ReduceOp,
        /// Optional axis (`None` = global).
        axis: Option<usize>,
        /// Whether to retain reduced dimensions as size 1.
        keepdim: bool,
    },
}

/// A single directed node in a computational [`Graph`].
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Unique identifier within the owning graph.
    pub id: NodeId,
    /// Operator executed by this node.
    pub op: Op,
    /// Ordered input dependencies (edges from producer -> this node).
    pub inputs: Vec<NodeId>,
    /// Optional checkpoint hint for memory-bounded execution / autodiff.
    pub checkpoint: bool,
}

/// Directed Acyclic Graph (DAG) representing a tensor computation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
}

impl Graph {
    /// Creates an empty [`Graph`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all nodes in insertion order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Returns the marked graph output node IDs.
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Looks up a node by [`NodeId`].
    pub fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(id.0).ok_or(Error::IndexOutOfBounds {
            index: id.0,
            len: self.nodes.len(),
        })
    }

    /// Adds a runtime input placeholder node.
    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
    ) -> Result<NodeId> {
        self.push_node(
            Op::Input {
                name: name.into(),
                shape,
                dtype,
            },
            Vec::new(),
        )
    }

    /// Adds an embedded constant tensor node.
    pub fn add_constant(&mut self, shape: Shape, dtype: DType, values: Vec<f64>) -> Result<NodeId> {
        if values.len() != shape.numel() {
            return Err(Error::BufferLengthMismatch {
                expected: shape.numel(),
                actual: values.len(),
            });
        }
        self.push_node(
            Op::Constant {
                shape,
                dtype,
                values,
            },
            Vec::new(),
        )
    }

    /// Adds an operator node consuming `inputs`.
    ///
    /// Validates node limits, input arity, and that all input IDs exist in this graph.
    pub fn add_op(&mut self, op: Op, inputs: Vec<NodeId>) -> Result<NodeId> {
        validate_arity(&op, inputs.len())?;
        for &inp in &inputs {
            if inp.0 >= self.nodes.len() {
                return Err(Error::IndexOutOfBounds {
                    index: inp.0,
                    len: self.nodes.len(),
                });
            }
        }
        self.push_node(op, inputs)
    }

    /// Marks `id` as a graph output.
    pub fn mark_output(&mut self, id: NodeId) -> Result<()> {
        if id.0 >= self.nodes.len() {
            return Err(Error::IndexOutOfBounds {
                index: id.0,
                len: self.nodes.len(),
            });
        }
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
        Ok(())
    }

    /// Marks `id` as a gradient/execution checkpoint boundary.
    pub fn mark_checkpoint(&mut self, id: NodeId) -> Result<()> {
        let len = self.nodes.len();
        let node = self.nodes.get_mut(id.0).ok_or(Error::IndexOutOfBounds {
            index: id.0,
            len,
        })?;
        node.checkpoint = true;
        Ok(())
    }

    /// Low-level constructor used by deserialization and optimization passes.
    ///
    /// Validates all edges and verifies the resulting graph is strictly acyclic.
    pub(crate) fn from_raw(nodes: Vec<Node>, outputs: Vec<NodeId>) -> Result<Self> {
        if nodes.len() > MAX_GRAPH_NODES {
            return Err(Error::InvalidArgument {
                context: "Graph::from_raw",
                reason: format!(
                    "graph node count {} exceeds limit {MAX_GRAPH_NODES}",
                    nodes.len()
                ),
            });
        }
        let n = nodes.len();
        for (idx, node) in nodes.iter().enumerate() {
            if node.id.0 != idx {
                return Err(Error::InvalidArgument {
                    context: "Graph::from_raw",
                    reason: format!("node id {} does not match slot {idx}", node.id.0),
                });
            }
            validate_arity(&node.op, node.inputs.len())?;
            for &inp in &node.inputs {
                if inp.0 >= n {
                    return Err(Error::IndexOutOfBounds {
                        index: inp.0,
                        len: n,
                    });
                }
            }
        }
        for &out in &outputs {
            if out.0 >= n {
                return Err(Error::IndexOutOfBounds {
                    index: out.0,
                    len: n,
                });
            }
        }
        let g = Self { nodes, outputs };
        // Enforce DAG invariant (cycle detection)
        let _ = g.topological_sort()?;
        Ok(g)
    }

    /// Performs iterative topological sorting (Kahn's algorithm) with deterministic
    /// tie-breaking and strict cycle detection.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

        for node in &self.nodes {
            if let Some(deg) = in_degree.get_mut(node.id.0) {
                *deg = node.inputs.len();
            }
            for &inp in &node.inputs {
                if let Some(list) = consumers.get_mut(inp.0) {
                    list.push(node.id.0);
                }
            }
        }

        let mut queue = VecDeque::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(idx);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(NodeId(u));
            if let Some(adj) = consumers.get(u) {
                for &v in adj {
                    if let Some(deg) = in_degree.get_mut(v) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        if order.len() != n {
            return Err(Error::InvalidArgument {
                context: "Graph::topological_sort",
                reason: "cycle detected in computational graph".to_owned(),
            });
        }
        Ok(order)
    }

    fn push_node(&mut self, op: Op, inputs: Vec<NodeId>) -> Result<NodeId> {
        if self.nodes.len() >= MAX_GRAPH_NODES {
            return Err(Error::InvalidArgument {
                context: "Graph::push_node",
                reason: format!("graph node count exceeds limit {MAX_GRAPH_NODES}"),
            });
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            id,
            op,
            inputs,
            checkpoint: false,
        });
        Ok(id)
    }
}

fn validate_arity(op: &Op, arity: usize) -> Result<()> {
    let expected = match op {
        Op::Input { .. } | Op::Constant { .. } => 0,
        Op::Unary(_)
        | Op::FusedUnaryChain(_)
        | Op::Reshape(_)
        | Op::Transpose { .. }
        | Op::Reduce { .. } => 1,
        Op::Binary(_) | Op::MatMul => 2,
    };
    if arity != expected {
        return Err(Error::InvalidArgument {
            context: "Graph::validate_arity",
            reason: format!("expected {expected} inputs for op {op:?}, got {arity}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cyclic_graph() {
        let n0 = Node {
            id: NodeId(0),
            op: Op::Unary(UnaryOp::Neg),
            inputs: vec![NodeId(1)],
            checkpoint: false,
        };
        let n1 = Node {
            id: NodeId(1),
            op: Op::Unary(UnaryOp::Abs),
            inputs: vec![NodeId(0)],
            checkpoint: false,
        };
        let err = Graph::from_raw(vec![n0, n1], vec![NodeId(1)]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { .. }));
    }
}
