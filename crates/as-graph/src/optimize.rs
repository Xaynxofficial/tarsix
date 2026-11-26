//! Graph optimization passes: Dead-Code Elimination (DCE) and Unary Operation Fusion.

use std::collections::{HashMap, VecDeque};

use as_backend::UnaryOp;
use as_core::Result;

use crate::ir::{Graph, Node, NodeId, Op};

/// Summary statistics from running [`optimize_graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizationReport {
    /// Original node count.
    pub nodes_before: usize,
    /// Optimized node count.
    pub nodes_after: usize,
    /// Number of dead nodes eliminated.
    pub dead_nodes_removed: usize,
    /// Number of unary ops fused into chains.
    pub unary_ops_fused: usize,
}

/// Runs Dead-Code Elimination followed by Unary Operation Fusion on `graph`.
pub fn optimize_graph(graph: &Graph) -> Result<(Graph, OptimizationReport)> {
    let nodes_before = graph.nodes().len();

    // Pass 1: Dead-Code Elimination (DCE)
    let (dce_graph, dead_removed) = dead_code_elimination(graph)?;

    // Pass 2: Unary Operation Fusion
    let (fused_graph, fused_count) = fuse_unary_chains(&dce_graph)?;

    let report = OptimizationReport {
        nodes_before,
        nodes_after: fused_graph.nodes().len(),
        dead_nodes_removed: dead_removed,
        unary_ops_fused: fused_count,
    };

    Ok((fused_graph, report))
}

pub(crate) fn dead_code_elimination(graph: &Graph) -> Result<(Graph, usize)> {
    let nodes = graph.nodes();
    if graph.outputs().is_empty() {
        return Ok((graph.clone(), 0));
    }

    let mut live = vec![false; nodes.len()];
    let mut queue = VecDeque::new();
    for &out in graph.outputs() {
        if let Some(flag) = live.get_mut(out.0)
            && !*flag
        {
            *flag = true;
            queue.push_back(out.0);
        }
    }

    while let Some(id) = queue.pop_front() {
        if let Some(node) = nodes.get(id) {
            for &inp in &node.inputs {
                if let Some(flag) = live.get_mut(inp.0)
                    && !*flag
                {
                    *flag = true;
                    queue.push_back(inp.0);
                }
            }
        }
    }

    let mut remap: HashMap<usize, NodeId> = HashMap::new();
    let mut new_nodes = Vec::new();
    let mut dead_count = 0usize;

    for (old_idx, node) in nodes.iter().enumerate() {
        if !live.get(old_idx).copied().unwrap_or(false) {
            dead_count += 1;
            continue;
        }
        let new_id = NodeId(new_nodes.len());
        remap.insert(old_idx, new_id);
        let mapped_inputs = node
            .inputs
            .iter()
            .filter_map(|inp| remap.get(&inp.0).copied())
            .collect();
        new_nodes.push(Node {
            id: new_id,
            op: node.op.clone(),
            inputs: mapped_inputs,
            checkpoint: node.checkpoint,
        });
    }

    let new_outputs = graph
        .outputs()
        .iter()
        .filter_map(|out| remap.get(&out.0).copied())
        .collect();

    Ok((Graph::from_raw(new_nodes, new_outputs)?, dead_count))
}

pub(crate) fn fuse_unary_chains(graph: &Graph) -> Result<(Graph, usize)> {
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
        if let Some(flag) = is_output.get_mut(out.0) {
            *flag = true;
        }
    }

    // Track nodes absorbed into a downstream fused chain
    let mut absorbed = vec![false; n];
    let mut fused_ops: HashMap<usize, (NodeId, Vec<UnaryOp>)> = HashMap::new();
    let mut total_fused = 0usize;

    // Walk in topological order so chains collapse greedily
    let topo = graph.topological_sort()?;
    for id in topo {
        let Some(node) = nodes.get(id.0) else {
            continue;
        };
        let Some(cur_op) = extract_unary_ops(&node.op) else {
            continue;
        };
        let Some(&parent_id) = node.inputs.first() else {
            continue;
        };
        let Some(parent_node) = nodes.get(parent_id.0) else {
            continue;
        };

        // Only fuse if parent is a unary op, has a single consumer, is not an output,
        // and is not a checkpoint boundary.
        let parent_consumers = consumer_count.get(parent_id.0).copied().unwrap_or(0);
        let parent_is_out = is_output.get(parent_id.0).copied().unwrap_or(false);
        if parent_consumers == 1 && !parent_is_out && !parent_node.checkpoint {
            if let Some((root_input, mut prev_chain)) = fused_ops.remove(&parent_id.0) {
                if let Some(flag) = absorbed.get_mut(parent_id.0) {
                    *flag = true;
                }
                prev_chain.extend(cur_op);
                total_fused += 1;
                fused_ops.insert(id.0, (root_input, prev_chain));
            } else if let Some(mut parent_ops) = extract_unary_ops(&parent_node.op)
                && let Some(&root_input) = parent_node.inputs.first()
            {
                if let Some(flag) = absorbed.get_mut(parent_id.0) {
                    *flag = true;
                }
                parent_ops.extend(cur_op);
                total_fused += 1;
                fused_ops.insert(id.0, (root_input, parent_ops));
            }
        }
    }

    if total_fused == 0 {
        return Ok((graph.clone(), 0));
    }

    let mut remap: HashMap<usize, NodeId> = HashMap::new();
    let mut new_nodes = Vec::new();
    for (old_idx, node) in nodes.iter().enumerate() {
        if absorbed.get(old_idx).copied().unwrap_or(false) {
            continue;
        }
        let new_id = NodeId(new_nodes.len());
        remap.insert(old_idx, new_id);

        if let Some((root_inp, chain)) = fused_ops.get(&old_idx) {
            let mapped_inp = remap.get(&root_inp.0).copied().unwrap_or(*root_inp);
            new_nodes.push(Node {
                id: new_id,
                op: Op::FusedUnaryChain(chain.clone()),
                inputs: vec![mapped_inp],
                checkpoint: node.checkpoint,
            });
        } else {
            let mapped_inputs = node
                .inputs
                .iter()
                .filter_map(|inp| remap.get(&inp.0).copied())
                .collect();
            new_nodes.push(Node {
                id: new_id,
                op: node.op.clone(),
                inputs: mapped_inputs,
                checkpoint: node.checkpoint,
            });
        }
    }

    let new_outputs = graph
        .outputs()
        .iter()
        .filter_map(|out| remap.get(&out.0).copied())
        .collect();

    Ok((Graph::from_raw(new_nodes, new_outputs)?, total_fused))
}

fn extract_unary_ops(op: &Op) -> Option<Vec<UnaryOp>> {
    match op {
        Op::Unary(u) => Some(vec![*u]),
        Op::FusedUnaryChain(chain) => Some(chain.clone()),
        _ => None,
    }
}
