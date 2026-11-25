//! Execution profiling structures for [`crate::InferenceSession`].

use as_graph::OptimizationReport;

/// Timing and graph statistics captured for a single inference invocation.
#[derive(Debug, Clone)]
pub struct InferenceProfile {
    /// Total end-to-end execution duration in microseconds.
    pub total_duration_us: u128,
    /// Number of graph nodes executed.
    pub executed_nodes: usize,
    /// Optional optimization summary if graph optimization was enabled.
    pub optimization: Option<OptimizationReport>,
}

/// Cumulative runtime telemetry across invocations in a session.
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    /// Total inference calls completed.
    pub total_runs: u64,
    /// Cumulative latency in microseconds.
    pub cumulative_us: u128,
}

impl RunStats {
    /// Average latency per run in microseconds.
    pub fn mean_us(&self) -> f64 {
        if self.total_runs == 0 {
            0.0
        } else {
            (self.cumulative_us as f64) / (self.total_runs as f64)
        }
    }
}
