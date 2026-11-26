//! Modular Graph Compiler: [`CompilerPass`], [`PassPipeline`], [`OptLevel`], and [`CompiledGraph`].
//!
//! Optimization passes included:
//! - [`DeadCodeEliminationPass`]: removes nodes unreachable from graph outputs
//! - [`ConstantFoldingPass`]: evaluates constant subgraphs at compile time
//! - [`AlgebraicSimplifyPass`]: eliminates identity ops, double transposes, double negations, and redundant reshapes
//! - [`CsePass`]: Common Subexpression Elimination for identical pure operations
//! - [`UnaryFusionPass`]: collapses consecutive single-consumer unary operations into [`Op::FusedUnaryChain`]
//! - [`LinearActivationFusionPass`]: fuses `MatMul [+ Bias] [+ UnaryAct]` into [`Op::FusedLinear`]

use std::collections::HashMap;

use as_backend::{BinaryOp, ReduceOp, UnaryOp};
use as_core::{DType, Result};
use as_memory::MemoryPlan;

use crate::executor::{ExecutionPlan, execute_graph};
use crate::infer::{TensorMeta, build_memory_plan, infer_graph_metadata};
use crate::ir::{Graph, Node, NodeId, Op};
use crate::optimize::{dead_code_elimination, fuse_unary_chains};

/// Compiler optimization level controlling which passes run in [`PassPipeline::for_level`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// No graph rewrites (validates and plans only).
    O0,
    /// Dead-Code Elimination + Algebraic Simplification.
    O1,
    /// `O1` + Constant Folding + Common Subexpression Elimination (CSE) + Unary Fusion.
    #[default]
    O2,
    /// `O2` + Linear-Bias-Activation Fusion (`MatMul + Bias + Act` -> `FusedLinear`) + cleanup DCE.
    O3,
}

/// Statistics produced by a single [`CompilerPass`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassStats {
    /// Name of the pass.
    pub pass_name: &'static str,
    /// Node count before this pass.
    pub nodes_before: usize,
    /// Node count after this pass.
    pub nodes_after: usize,
    /// Number of rewrites/eliminations/fusions applied.
    pub rewrites: usize,
}

/// Full compilation summary produced by [`PassPipeline::run`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationReport {
    /// Optimization level used.
    pub opt_level: OptLevel,
    /// Initial node count before any pass.
    pub nodes_before: usize,
    /// Final node count after all passes.
    pub nodes_after: usize,
    /// Ordered per-pass telemetry.
    pub pass_stats: Vec<PassStats>,
}

/// A modular graph transformation pass.
pub trait CompilerPass: Send + Sync {
    /// Static human-readable name of this pass.
    fn name(&self) -> &'static str;

    /// Runs the pass on `graph`, returning `(transformed_graph, rewrite_count)`.
    fn run(&self, graph: &Graph) -> Result<(Graph, usize)>;
}

/// Ordered sequence of [`CompilerPass`] instances.
pub struct PassPipeline {
    opt_level: OptLevel,
    passes: Vec<Box<dyn CompilerPass>>,
}

impl PassPipeline {
    /// Creates an empty [`PassPipeline`].
    pub fn empty(opt_level: OptLevel) -> Self {
        Self {
            opt_level,
            passes: Vec::new(),
        }
    }

    /// Appends a pass to the pipeline.
    pub fn push(&mut self, pass: Box<dyn CompilerPass>) -> &mut Self {
        self.passes.push(pass);
        self
    }

    /// Constructs the standard Tarsix pass pipeline for `level`.
    pub fn for_level(level: OptLevel) -> Self {
        let mut p = Self::empty(level);
        match level {
            OptLevel::O0 => {}
            OptLevel::O1 => {
                p.push(Box::new(AlgebraicSimplifyPass))
                    .push(Box::new(DeadCodeEliminationPass));
            }
            OptLevel::O2 => {
                p.push(Box::new(ConstantFoldingPass))
                    .push(Box::new(AlgebraicSimplifyPass))
                    .push(Box::new(CsePass))
                    .push(Box::new(DeadCodeEliminationPass))
                    .push(Box::new(UnaryFusionPass));
            }
            OptLevel::O3 => {
                p.push(Box::new(ConstantFoldingPass))
                    .push(Box::new(AlgebraicSimplifyPass))
                    .push(Box::new(CsePass))
                    .push(Box::new(DeadCodeEliminationPass))
                    .push(Box::new(LinearActivationFusionPass))
                    .push(Box::new(UnaryFusionPass))
                    .push(Box::new(DeadCodeEliminationPass));
            }
        }
        p
    }

    /// Executes all passes in order and returns the optimized [`Graph`] and [`CompilationReport`].
    pub fn run(&self, graph: &Graph) -> Result<(Graph, CompilationReport)> {
        let nodes_before = graph.nodes().len();
        let mut current = graph.clone();
        let mut pass_stats = Vec::with_capacity(self.passes.len());

        for pass in &self.passes {
            let before = current.nodes().len();
            let (next, rewrites) = pass.run(&current)?;
            let after = next.nodes().len();
            pass_stats.push(PassStats {
                pass_name: pass.name(),
                nodes_before: before,
                nodes_after: after,
                rewrites,
            });
            current = next;
        }

        let nodes_after = current.nodes().len();
        Ok((
            current,
            CompilationReport {
                opt_level: self.opt_level,
                nodes_before,
                nodes_after,
                pass_stats,
            },
        ))
    }
}

/// Fully compiled graph artifact ready for scheduled execution.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// Optimized computation graph.
    pub graph: Graph,
    /// Topological execution schedule and liveness reference counts.
    pub execution_plan: ExecutionPlan,
    /// Statically inferred shapes and dtypes for every node.
    pub metadata: Vec<TensorMeta>,
    /// Static interval-coloring buffer reuse plan from `as-memory`.
    pub memory_plan: MemoryPlan,
    /// Compiler pass report.
    pub report: CompilationReport,
}

/// Runs the [`PassPipeline`] for `level`, performs static shape/dtype inference,
/// compiles the [`ExecutionPlan`], and builds a static [`MemoryPlan`].
pub fn compile_graph(graph: &Graph, level: OptLevel) -> Result<CompiledGraph> {
    let pipeline = PassPipeline::for_level(level);
    let (opt_graph, report) = pipeline.run(graph)?;
    let metadata = infer_graph_metadata(&opt_graph)?;
    let execution_plan = ExecutionPlan::compile(&opt_graph)?;
    let memory_plan = build_memory_plan(&opt_graph, &execution_plan, &metadata)?;
    Ok(CompiledGraph {
        graph: opt_graph,
        execution_plan,
        metadata,
        memory_plan,
        report,
    })
}

/// Pass 1: Dead-Code Elimination.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeadCodeEliminationPass;

impl CompilerPass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dead_code_elimination"
    }

    fn run(&self, graph: &Graph) -> Result<(Graph, usize)> {
        dead_code_elimination(graph)
    }
}

/// Pass 2: Unary Operation Chain Fusion.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnaryFusionPass;

impl CompilerPass for UnaryFusionPass {
    fn name(&self) -> &'static str {
        "unary_fusion"
    }

    fn run(&self, graph: &Graph) -> Result<(Graph, usize)> {
        fuse_unary_chains(graph)
    }
}

/// Pass 3: Compile-Time Constant Folding.
///
/// Any pure operation whose inputs are all [`Op::Constant`] is evaluated once at compile
/// time and replaced by a single [`Op::Constant`] node.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConstantFoldingPass;

impl CompilerPass for ConstantFoldingPass {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn run(&self, graph: &Graph) -> Result<(Graph, usize)> {
        let topo = graph.topological_sort()?;
        let mut nodes = graph.nodes().to_vec();
        let mut folded_count = 0usize;

        for node_id in topo {
            let Some(node) = nodes.get(node_id.0).cloned() else {
                continue;
            };
            if matches!(node.op, Op::Input { .. } | Op::Constant { .. }) || node.inputs.is_empty() {
                continue;
            }
            let all_const = node.inputs.iter().all(|&inp| {
                nodes
                    .get(inp.0)
                    .is_some_and(|n| matches!(n.op, Op::Constant { .. }))
            });
            if !all_const {
                continue;
            }

            // Build a tiny isolated subgraph for this node + its constant inputs and evaluate it
            let mut sub_nodes = Vec::with_capacity(node.inputs.len() + 1);
            let mut sub_inputs = Vec::with_capacity(node.inputs.len());
            for &inp_id in &node.inputs {
                let Some(inp_node) = nodes.get(inp_id.0) else {
                    continue;
                };
                let sid = NodeId(sub_nodes.len());
                sub_nodes.push(Node {
                    id: sid,
                    op: inp_node.op.clone(),
                    inputs: Vec::new(),
                    checkpoint: false,
                });
                sub_inputs.push(sid);
            }
            let target_sid = NodeId(sub_nodes.len());
            sub_nodes.push(Node {
                id: target_sid,
                op: node.op.clone(),
                inputs: sub_inputs,
                checkpoint: false,
            });

            let sub_graph = Graph::from_raw(sub_nodes, vec![target_sid])?;
            let out_map = execute_graph(&sub_graph, &HashMap::new())?;
            let Some(out_tensor) = out_map.get(&target_sid) else {
                continue;
            };
            let dtype = out_tensor.dtype();
            let shape = out_tensor.shape().clone();
            let values = out_tensor.to_dtype(DType::F64)?.to_f64_vec()?;

            if let Some(slot) = nodes.get_mut(node_id.0) {
                slot.op = Op::Constant {
                    shape,
                    dtype,
                    values,
                };
                slot.inputs.clear();
                folded_count += 1;
            }
        }

        if folded_count == 0 {
            return Ok((graph.clone(), 0));
        }
        Ok((Graph::from_raw(nodes, graph.outputs().to_vec())?, folded_count))
    }
}

/// Pass 4: Algebraic Simplification.
///
/// Simplifies:
/// - `Transpose(Transpose(x, a, b), a, b)` -> `x`
/// - `Reshape(Reshape(x, _), s2)` -> `Reshape(x, s2)`
/// - `Neg(Neg(x))` -> `x`
/// - `Add(x, 0)`, `Add(0, x)`, `Sub(x, 0)`, `Mul(x, 1)`, `Mul(1, x)`, `Div(x, 1)` -> `x`
#[derive(Debug, Default, Clone, Copy)]
pub struct AlgebraicSimplifyPass;

impl CompilerPass for AlgebraicSimplifyPass {
    fn name(&self) -> &'static str {
        "algebraic_simplify"
    }

    fn run(&self, graph: &Graph) -> Result<(Graph, usize)> {
        let nodes = graph.nodes();
        let topo = graph.topological_sort()?;
        let mut redirect: Vec<NodeId> = (0..nodes.len()).map(NodeId).collect();
        let mut rewritten_nodes = nodes.to_vec();
        let mut rewrites = 0usize;

        for node_id in topo {
            let Some(mut node) = rewritten_nodes.get(node_id.0).cloned() else {
                continue;
            };
            for inp in &mut node.inputs {
                *inp = resolve_redirect(&redirect, *inp);
            }

            // Check algebraic identities
            let replacement: Option<NodeId> = match &node.op {
                Op::Transpose { axis_a, axis_b } => {
                    if *axis_a == *axis_b {
                        node.inputs.first().copied()
                    } else if let Some(&inp_id) = node.inputs.first()
                        && let Some(parent) = rewritten_nodes.get(inp_id.0)
                        && let Op::Transpose {
                            axis_a: pa,
                            axis_b: pb,
                        } = &parent.op
                        && ((*pa == *axis_a && *pb == *axis_b)
                            || (*pa == *axis_b && *pb == *axis_a))
                    {
                        parent.inputs.first().copied()
                    } else {
                        None
                    }
                }
                Op::Unary(UnaryOp::Neg) => {
                    if let Some(&inp_id) = node.inputs.first()
                        && let Some(parent) = rewritten_nodes.get(inp_id.0)
                        && matches!(parent.op, Op::Unary(UnaryOp::Neg))
                    {
                        parent.inputs.first().copied()
                    } else {
                        None
                    }
                }
                Op::Binary(bin_op) => {
                    let lhs_id = node.inputs.first().copied();
                    let rhs_id = node.inputs.get(1).copied();
                    match (bin_op, lhs_id, rhs_id) {
                        (BinaryOp::Add, Some(l), Some(r)) => {
                            if is_all_const_value(&rewritten_nodes, r, 0.0) {
                                Some(l)
                            } else if is_all_const_value(&rewritten_nodes, l, 0.0) {
                                Some(r)
                            } else {
                                None
                            }
                        }
                        (BinaryOp::Sub, Some(l), Some(r)) => {
                            if is_all_const_value(&rewritten_nodes, r, 0.0) {
                                Some(l)
                            } else {
                                None
                            }
                        }
                        (BinaryOp::Mul, Some(l), Some(r)) => {
                            if is_all_const_value(&rewritten_nodes, r, 1.0) {
                                Some(l)
                            } else if is_all_const_value(&rewritten_nodes, l, 1.0) {
                                Some(r)
                            } else {
                                None
                            }
                        }
                        (BinaryOp::Div, Some(l), Some(r)) => {
                            if is_all_const_value(&rewritten_nodes, r, 1.0) {
                                Some(l)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            };

            if let Some(target_id) = replacement {
                let resolved = resolve_redirect(&redirect, target_id);
                if let Some(slot) = redirect.get_mut(node_id.0) {
                    *slot = resolved;
                }
                rewrites += 1;
                continue;
            }

            // Collapse Reshape(Reshape(x, _), target) -> Reshape(x, target)
            if matches!(node.op, Op::Reshape(_))
                && let Some(&inp_id) = node.inputs.first()
                && let Some(parent) = rewritten_nodes.get(inp_id.0)
                && matches!(parent.op, Op::Reshape(_))
                && let Some(&root) = parent.inputs.first()
            {
                node.inputs = vec![resolve_redirect(&redirect, root)];
                rewrites += 1;
            }

            if let Some(slot) = rewritten_nodes.get_mut(node_id.0) {
                *slot = node;
            }
        }

        if rewrites == 0 {
            return Ok((graph.clone(), 0));
        }

        let new_outputs: Vec<NodeId> = graph
            .outputs()
            .iter()
            .map(|&o| resolve_redirect(&redirect, o))
            .collect();
        let rebuilt = Graph::from_raw(rewritten_nodes, new_outputs)?;
        Ok((rebuilt, rewrites))
    }
}

/// Pass 5: Common Subexpression Elimination (CSE).
///
/// Deduplicates pure operations with identical operator semantics and input dependencies.
#[derive(Debug, Default, Clone, Copy)]
pub struct CsePass;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CseOpKey {
    Binary(u8),
    Unary(u8, u64),
    FusedUnary(Vec<(u8, u64)>),
    MatMul,
    Reshape(Vec<usize>),
    Transpose(usize, usize),
    Reduce(u8, Option<usize>, bool),
    FusedLinear(bool, Option<(u8, u64)>),
}

impl CompilerPass for CsePass {
    fn name(&self) -> &'static str {
        "common_subexpression_elimination"
    }

    fn run(&self, graph: &Graph) -> Result<(Graph, usize)> {
        let topo = graph.topological_sort()?;
        let mut nodes = graph.nodes().to_vec();
        let mut redirect: Vec<NodeId> = (0..nodes.len()).map(NodeId).collect();
        let mut seen: HashMap<(CseOpKey, Vec<NodeId>), NodeId> = HashMap::new();
        let mut eliminated = 0usize;

        for node_id in topo {
            let Some(mut node) = nodes.get(node_id.0).cloned() else {
                continue;
            };
            for inp in &mut node.inputs {
                *inp = resolve_redirect(&redirect, *inp);
            }

            if let Some(op_key) = make_cse_key(&node.op) {
                let key = (op_key, node.inputs.clone());
                if let Some(&existing_id) = seen.get(&key) {
                    if let Some(slot) = redirect.get_mut(node_id.0) {
                        *slot = existing_id;
                    }
                    eliminated += 1;
                    continue;
                }
                seen.insert(key, node_id);
            }

            if let Some(slot) = nodes.get_mut(node_id.0) {
                *slot = node;
            }
        }

        if eliminated == 0 {
            return Ok((graph.clone(), 0));
        }

        let new_outputs: Vec<NodeId> = graph
            .outputs()
            .iter()
            .map(|&o| resolve_redirect(&redirect, o))
            .collect();
        Ok((Graph::from_raw(nodes, new_outputs)?, eliminated))
    }
}

/// Pass 6: Linear + Bias + Activation Fusion (`MatMul [+ Add(bias)] [+ UnaryAct]` -> [`Op::FusedLinear`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct LinearActivationFusionPass;

impl CompilerPass for LinearActivationFusionPass {
    fn name(&self) -> &'static str {
        "linear_activation_fusion"
    }

    fn run(&self, graph: &Graph) -> Result<(Graph, usize)> {
        let nodes = graph.nodes();
        let n = nodes.len();
        let mut consumer_count = vec![0usize; n];
        for node in nodes {
            for &inp in &node.inputs {
                if let Some(c) = consumer_count.get_mut(inp.0) {
                    *c += 1;
                }
            }
        }
        let mut is_output = vec![false; n];
        for &out in graph.outputs() {
            if let Some(f) = is_output.get_mut(out.0) {
                *f = true;
            }
        }

        let mut new_nodes = nodes.to_vec();
        let mut fused = 0usize;
        let topo = graph.topological_sort()?;

        // Step A: Fuse MatMul + Add(bias) -> FusedLinear { has_bias: true, activation: None }
        for &id in &topo {
            let Some(node) = new_nodes.get(id.0).cloned() else {
                continue;
            };
            if !matches!(node.op, Op::Binary(BinaryOp::Add)) || node.inputs.len() != 2 {
                continue;
            }
            let (Some(&lhs), Some(&rhs)) = (node.inputs.first(), node.inputs.get(1)) else {
                continue;
            };

            let candidate = if can_absorb_matmul(&new_nodes, &consumer_count, &is_output, lhs) {
                Some((lhs, rhs))
            } else if can_absorb_matmul(&new_nodes, &consumer_count, &is_output, rhs) {
                Some((rhs, lhs))
            } else {
                None
            };

            if let Some((mm_id, bias_id)) = candidate
                && let Some(mm_node) = new_nodes.get(mm_id.0).cloned()
                && mm_node.inputs.len() == 2
                && let (Some(&x), Some(&w)) = (mm_node.inputs.first(), mm_node.inputs.get(1))
                && let Some(slot) = new_nodes.get_mut(id.0)
            {
                slot.op = Op::FusedLinear {
                    has_bias: true,
                    activation: None,
                };
                slot.inputs = vec![x, w, bias_id];
                fused += 1;
            }
        }

        // Step B: Fuse trailing single Unary activation into MatMul or FusedLinear(activation: None)
        for &id in &topo {
            let Some(node) = new_nodes.get(id.0).cloned() else {
                continue;
            };
            let Op::Unary(act) = node.op else {
                continue;
            };
            let Some(&parent_id) = node.inputs.first() else {
                continue;
            };
            let parent_consumers = consumer_count.get(parent_id.0).copied().unwrap_or(0);
            let parent_is_out = is_output.get(parent_id.0).copied().unwrap_or(false);
            if parent_consumers != 1 || parent_is_out {
                continue;
            }
            let Some(parent) = new_nodes.get(parent_id.0).cloned() else {
                continue;
            };
            if parent.checkpoint {
                continue;
            }

            match parent.op {
                Op::MatMul => {
                    if let Some(slot) = new_nodes.get_mut(id.0) {
                        slot.op = Op::FusedLinear {
                            has_bias: false,
                            activation: Some(act),
                        };
                        slot.inputs = parent.inputs;
                        fused += 1;
                    }
                }
                Op::FusedLinear {
                    has_bias,
                    activation: None,
                } => {
                    if let Some(slot) = new_nodes.get_mut(id.0) {
                        slot.op = Op::FusedLinear {
                            has_bias,
                            activation: Some(act),
                        };
                        slot.inputs = parent.inputs;
                        fused += 1;
                    }
                }
                _ => {}
            }
        }

        if fused == 0 {
            return Ok((graph.clone(), 0));
        }
        Ok((Graph::from_raw(new_nodes, graph.outputs().to_vec())?, fused))
    }
}

fn can_absorb_matmul(
    nodes: &[Node],
    consumer_count: &[usize],
    is_output: &[bool],
    id: NodeId,
) -> bool {
    let consumers = consumer_count.get(id.0).copied().unwrap_or(0);
    let out = is_output.get(id.0).copied().unwrap_or(false);
    if consumers != 1 || out {
        return false;
    }
    nodes
        .get(id.0)
        .is_some_and(|n| matches!(n.op, Op::MatMul) && !n.checkpoint)
}

fn resolve_redirect(redirect: &[NodeId], mut id: NodeId) -> NodeId {
    for _ in 0..redirect.len() {
        let next = redirect.get(id.0).copied().unwrap_or(id);
        if next == id {
            return id;
        }
        id = next;
    }
    id
}

fn is_all_const_value(nodes: &[Node], id: NodeId, target: f64) -> bool {
    let Some(node) = nodes.get(id.0) else {
        return false;
    };
    match &node.op {
        Op::Constant { values, shape, .. } => {
            // Only fold scalar-broadcastable [1] constants safely without changing output shape
            shape.numel() == 1 && values.iter().all(|&v| v == target)
        }
        _ => false,
    }
}

fn unary_key(op: UnaryOp) -> (u8, u64) {
    match op {
        UnaryOp::Neg => (0, 0),
        UnaryOp::Abs => (1, 0),
        UnaryOp::Exp => (2, 0),
        UnaryOp::Ln => (3, 0),
        UnaryOp::Sqrt => (4, 0),
        UnaryOp::Relu => (5, 0),
        UnaryOp::Powf(p) => (6, p.to_bits()),
    }
}

fn make_cse_key(op: &Op) -> Option<CseOpKey> {
    match op {
        Op::Input { .. } | Op::Constant { .. } => None,
        Op::Binary(b) => Some(CseOpKey::Binary(match b {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
        })),
        Op::Unary(u) => {
            let (tag, bits) = unary_key(*u);
            Some(CseOpKey::Unary(tag, bits))
        }
        Op::FusedUnaryChain(chain) => {
            Some(CseOpKey::FusedUnary(chain.iter().copied().map(unary_key).collect()))
        }
        Op::MatMul => Some(CseOpKey::MatMul),
        Op::Reshape(s) => Some(CseOpKey::Reshape(s.dims().to_vec())),
        Op::Transpose { axis_a, axis_b } => Some(CseOpKey::Transpose(*axis_a, *axis_b)),
        Op::Reduce { op, axis, keepdim } => Some(CseOpKey::Reduce(
            match op {
                ReduceOp::Sum => 0,
                ReduceOp::Mean => 1,
                ReduceOp::Max => 2,
                ReduceOp::Min => 3,
            },
            *axis,
            *keepdim,
        )),
        Op::FusedLinear {
            has_bias,
            activation,
        } => Some(CseOpKey::FusedLinear(
            *has_bias,
            activation.map(unary_key),
        )),
    }
}
