//! Parallel DAG wave partitioning (ASAP level scheduling).
//!
//! Groups independent nodes that have all dependencies satisfied in earlier waves
//! into the same [`ExecutionWave`] so they can be dispatched concurrently.

use as_core::Result;
use as_graph::{Graph, NodeId};

/// A set of mutually independent graph nodes that can execute concurrently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionWave {
    /// Zero-based wave level index.
    pub wave_index: usize,
    /// Nodes scheduled in this wave.
    pub nodes: Vec<NodeId>,
}

/// Complete wave schedule for a computation [`Graph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveSchedule {
    /// Ordered execution waves.
    pub waves: Vec<ExecutionWave>,
    /// Critical path depth (number of sequential waves).
    pub critical_path_depth: usize,
    /// Maximum number of concurrent nodes in any single wave.
    pub max_parallelism: usize,
}

/// Partitions `graph` into ASAP parallel execution waves.
pub fn build_wave_schedule(graph: &Graph) -> Result<WaveSchedule> {
    let topo = graph.topological_sort()?;
    let n = graph.nodes().len();
    let mut level = vec![0usize; n];
    let mut max_level = 0usize;

    for &node_id in &topo {
        let node = graph.node(node_id)?;
        let mut my_level = 0usize;
        for &inp in &node.inputs {
            let dep_level = level.get(inp.0).copied().unwrap_or(0);
            my_level = my_level.max(dep_level + 1);
        }
        if let Some(slot) = level.get_mut(node_id.0) {
            *slot = my_level;
        }
        max_level = max_level.max(my_level);
    }

    if topo.is_empty() {
        return Ok(WaveSchedule {
            waves: Vec::new(),
            critical_path_depth: 0,
            max_parallelism: 0,
        });
    }

    let num_waves = max_level + 1;
    let mut buckets: Vec<Vec<NodeId>> = vec![Vec::new(); num_waves];
    for node_id in topo {
        let w = level.get(node_id.0).copied().unwrap_or(0);
        if let Some(bucket) = buckets.get_mut(w) {
            bucket.push(node_id);
        }
    }

    let mut max_parallelism = 0usize;
    let mut waves = Vec::with_capacity(num_waves);
    for (wave_index, nodes) in buckets.into_iter().enumerate() {
        max_parallelism = max_parallelism.max(nodes.len());
        waves.push(ExecutionWave { wave_index, nodes });
    }

    Ok(WaveSchedule {
        critical_path_depth: waves.len(),
        max_parallelism,
        waves,
    })
}
