//! Deep execution profiling, roofline bottleneck analysis, Chrome Trace JSON,
//! and SVG timeline visualization for [`crate::InferenceSession`].

use std::fmt::Write as _;

use as_core::DType;
use as_graph::{CompilationReport, NodeExecutionRecord, NodeId, OptimizationReport};
use as_memory::MemoryPlan;

/// Roofline bottleneck classification for a single operator execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckKind {
    /// High arithmetic intensity (`FLOPs / Byte > 16.0`), typically GEMM / FusedLinear.
    ComputeBound,
    /// Low arithmetic intensity (`0 < FLOPs / Byte <= 16.0`), typically elementwise / reductions.
    MemoryBound,
    /// Zero-FLOP structural or I/O operation (Input, Constant, Reshape, Transpose).
    Overhead,
}

impl BottleneckKind {
    /// Human-readable label for tables and traces.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ComputeBound => "compute-bound",
            Self::MemoryBound => "memory-bound",
            Self::Overhead => "overhead",
        }
    }
}

/// Detailed performance and memory telemetry for a single executed graph node.
#[derive(Debug, Clone)]
pub struct OpProfile {
    /// Executed node ID.
    pub node_id: NodeId,
    /// Static operator name.
    pub op_name: &'static str,
    /// Start offset from inference start in microseconds.
    pub start_us: u128,
    /// Execution duration in microseconds.
    pub duration_us: u128,
    /// Input tensor shapes.
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Output tensor dtype.
    pub output_dtype: DType,
    /// Estimated floating-point operations.
    pub estimated_flops: u64,
    /// Estimated bytes read from inputs.
    pub read_bytes: usize,
    /// Estimated bytes written to output.
    pub write_bytes: usize,
    /// Live bytes held immediately after this operator completed.
    pub live_bytes_after: usize,
    /// Arithmetic intensity (`FLOPs / total_bytes`).
    pub arithmetic_intensity: f64,
    /// Achieved compute throughput in GFLOP/s.
    pub achieved_gflops: f64,
    /// Achieved memory bandwidth in GB/s.
    pub achieved_bandwidth_gbs: f64,
    /// Roofline bottleneck classification.
    pub bottleneck: BottleneckKind,
}

impl OpProfile {
    /// Constructs an [`OpProfile`] from a raw [`NodeExecutionRecord`].
    pub fn from_record(rec: NodeExecutionRecord) -> Self {
        let total_bytes = rec.read_bytes.saturating_add(rec.write_bytes);
        let arithmetic_intensity = if total_bytes == 0 {
            0.0
        } else {
            (rec.estimated_flops as f64) / (total_bytes as f64)
        };
        let dur_sec = (rec.duration_us.max(1) as f64) * 1e-6;
        let achieved_gflops = (rec.estimated_flops as f64) / dur_sec / 1e9;
        let achieved_bandwidth_gbs = (total_bytes as f64) / dur_sec / 1e9;
        let bottleneck = if rec.estimated_flops == 0 {
            BottleneckKind::Overhead
        } else if arithmetic_intensity > 16.0 {
            BottleneckKind::ComputeBound
        } else {
            BottleneckKind::MemoryBound
        };

        Self {
            node_id: rec.node_id,
            op_name: rec.op_name,
            start_us: rec.start_us,
            duration_us: rec.duration_us,
            input_shapes: rec.input_shapes,
            output_shape: rec.output_shape,
            output_dtype: rec.output_dtype,
            estimated_flops: rec.estimated_flops,
            read_bytes: rec.read_bytes,
            write_bytes: rec.write_bytes,
            live_bytes_after: rec.live_bytes_after,
            arithmetic_intensity,
            achieved_gflops,
            achieved_bandwidth_gbs,
            bottleneck,
        }
    }
}

/// Comprehensive timing, FLOPs, memory, and compiler statistics for a single inference run.
#[derive(Debug, Clone)]
pub struct InferenceProfile {
    /// Total end-to-end execution duration in microseconds.
    pub total_duration_us: u128,
    /// Number of graph nodes executed.
    pub executed_nodes: usize,
    /// Optional legacy optimization summary if graph optimization was enabled.
    pub optimization: Option<OptimizationReport>,
    /// Optional full compiler pass pipeline report.
    pub compilation: Option<CompilationReport>,
    /// Optional static buffer-reuse memory plan.
    pub memory_plan: Option<MemoryPlan>,
    /// Per-operator execution telemetry in topological order.
    pub op_profiles: Vec<OpProfile>,
    /// Peak live tensor bytes observed during execution.
    pub peak_live_bytes: usize,
    /// Total estimated FLOPs across all executed operators.
    pub total_flops: u64,
    /// Total memory traffic (bytes read + bytes written) across all operators.
    pub total_memory_bytes: usize,
}

impl InferenceProfile {
    /// Builds an [`InferenceProfile`] from raw execution records and compiler metadata.
    pub fn from_records(
        total_duration_us: u128,
        records: Vec<NodeExecutionRecord>,
        optimization: Option<OptimizationReport>,
        compilation: Option<CompilationReport>,
        memory_plan: Option<MemoryPlan>,
    ) -> Self {
        let executed_nodes = records.len();
        let mut peak_live_bytes = 0usize;
        let mut total_flops = 0u64;
        let mut total_memory_bytes = 0usize;
        let mut op_profiles = Vec::with_capacity(executed_nodes);

        for rec in records {
            peak_live_bytes = peak_live_bytes.max(rec.live_bytes_after);
            total_flops = total_flops.saturating_add(rec.estimated_flops);
            total_memory_bytes = total_memory_bytes
                .saturating_add(rec.read_bytes)
                .saturating_add(rec.write_bytes);
            op_profiles.push(OpProfile::from_record(rec));
        }

        Self {
            total_duration_us,
            executed_nodes,
            optimization,
            compilation,
            memory_plan,
            op_profiles,
            peak_live_bytes,
            total_flops,
            total_memory_bytes,
        }
    }

    /// End-to-end achieved compute throughput in GFLOP/s.
    pub fn achieved_gflops(&self) -> f64 {
        let sec = (self.total_duration_us.max(1) as f64) * 1e-6;
        (self.total_flops as f64) / sec / 1e9
    }

    /// End-to-end achieved memory bandwidth in GB/s.
    pub fn achieved_bandwidth_gbs(&self) -> f64 {
        let sec = (self.total_duration_us.max(1) as f64) * 1e-6;
        (self.total_memory_bytes as f64) / sec / 1e9
    }

    /// Returns the `top_k` slowest operators sorted by descending duration.
    pub fn slowest_ops(&self, top_k: usize) -> Vec<&OpProfile> {
        let mut refs: Vec<&OpProfile> = self.op_profiles.iter().collect();
        refs.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));
        refs.truncate(top_k);
        refs
    }

    /// Renders a clean Markdown/CLI summary table of operator latency, FLOPs, and bottlenecks.
    pub fn to_summary_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "=== Tarsix Model Profiler Summary ===\nTotal: {} us | Nodes: {} | FLOPs: {} ({:.3} GFLOP/s) | Peak Live: {} B",
            self.total_duration_us,
            self.executed_nodes,
            self.total_flops,
            self.achieved_gflops(),
            self.peak_live_bytes
        );
        if let Some(mp) = &self.memory_plan {
            let _ = writeln!(
                out,
                "Static Memory Plan: {} slots | Planned Peak: {} B vs Naive Total: {} B ({:.1}% saved)",
                mp.slots.len(),
                mp.planned_peak_bytes,
                mp.naive_total_bytes,
                mp.memory_savings_ratio() * 100.0
            );
        }
        let _ = writeln!(
            out,
            "| Node | Op | Time (us) | Share | Output Shape | FLOPs | Traffic (B) | Bottleneck |\n|---:|:---|---:|---:|:---|---:|---:|:---|"
        );
        let denom = self.total_duration_us.max(1) as f64;
        for op in &self.op_profiles {
            let share = (op.duration_us as f64) * 100.0 / denom;
            let traffic = op.read_bytes.saturating_add(op.write_bytes);
            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.1}% | {:?} | {} | {} | {} |",
                op.node_id.0,
                op.op_name,
                op.duration_us,
                share,
                op.output_shape,
                op.estimated_flops,
                traffic,
                op.bottleneck.as_str()
            );
        }
        out
    }

    /// Exports per-operator execution spans in Chrome Trace Event Format (`chrome://tracing` / Perfetto).
    pub fn to_chrome_trace_json(&self) -> String {
        let mut out = String::from("{\"traceEvents\":[\n");
        for (i, op) in self.op_profiles.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            let _ = write!(
                out,
                "  {{\"name\":\"{}#{}\",\"cat\":\"{}\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":1,\"tid\":1,\"args\":{{\"flops\":{},\"read_bytes\":{},\"write_bytes\":{},\"live_bytes\":{},\"ai\":{:.3}}}}}",
                op.op_name,
                op.node_id.0,
                op.bottleneck.as_str(),
                op.start_us,
                op.duration_us.max(1),
                op.estimated_flops,
                op.read_bytes,
                op.write_bytes,
                op.live_bytes_after,
                op.arithmetic_intensity
            );
        }
        out.push_str("\n]}\n");
        out
    }

    /// Renders a self-contained, zero-dependency SVG timeline chart color-coded by roofline bottleneck.
    pub fn to_svg_flamegraph(&self) -> String {
        let width: f64 = 960.0;
        let pad_left: f64 = 160.0;
        let pad_right: f64 = 30.0;
        let row_h: f64 = 24.0;
        let top_margin: f64 = 52.0;
        let chart_w = width - pad_left - pad_right;
        let height = top_margin + (self.op_profiles.len().max(1) as f64) * row_h + 36.0;

        let total_span = self
            .op_profiles
            .iter()
            .map(|o| o.start_us.saturating_add(o.duration_us.max(1)))
            .max()
            .unwrap_or(1)
            .max(self.total_duration_us.max(1)) as f64;

        let mut svg = String::new();
        let _ = writeln!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        );
        svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#0f172a\"/>\n");
        let _ = writeln!(
            svg,
            "<text x=\"16\" y=\"28\" fill=\"#f8fafc\" font-family=\"monospace\" font-size=\"15\" font-weight=\"bold\">Tarsix Model Profiler Timeline ({} us | {} ops | Peak {} B)</text>",
            self.total_duration_us, self.executed_nodes, self.peak_live_bytes
        );
        svg.push_str(
            "<text x=\"16\" y=\"44\" fill=\"#94a3b8\" font-family=\"monospace\" font-size=\"11\">Legend: [Orange] Compute-Bound  [Sky] Memory-Bound  [Slate] Overhead</text>\n",
        );

        for (idx, op) in self.op_profiles.iter().enumerate() {
            let y = top_margin + (idx as f64) * row_h;
            let x = pad_left + ((op.start_us as f64) / total_span) * chart_w;
            let bar_w = (((op.duration_us.max(1) as f64) / total_span) * chart_w).max(3.0);
            let color = match op.bottleneck {
                BottleneckKind::ComputeBound => "#f97316",
                BottleneckKind::MemoryBound => "#38bdf8",
                BottleneckKind::Overhead => "#64748b",
            };
            let _ = writeln!(
                svg,
                "<text x=\"12\" y=\"{:.1}\" fill=\"#e2e8f0\" font-family=\"monospace\" font-size=\"11\">#{} {}</text>",
                y + 15.0,
                op.node_id.0,
                op.op_name
            );
            let _ = writeln!(
                svg,
                "<rect x=\"{x:.1}\" y=\"{:.1}\" width=\"{bar_w:.1}\" height=\"16\" rx=\"3\" fill=\"{color}\"><title>{}#{}: {} us, {} FLOPs, {}</title></rect>",
                y + 3.0,
                op.op_name,
                op.node_id.0,
                op.duration_us,
                op.estimated_flops,
                op.bottleneck.as_str()
            );
        }

        svg.push_str("</svg>\n");
        svg
    }
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
