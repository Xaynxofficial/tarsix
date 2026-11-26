//! Integration tests for the Phase 1 Model Profiler (op-level telemetry,
//! Roofline classification, Chrome Trace JSON, and SVG timeline export).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;

use as_backend::{BinaryOp, UnaryOp};
use as_core::{DType, Shape};
use as_graph::{Graph, Op};
use as_runtime::{InferenceSession, OptLevel, RuntimeConfig};
use as_tensor::Tensor;

#[test]
fn profiler_captures_op_telemetry_chrome_trace_and_svg() {
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![8, 16]).unwrap(), DType::F32).unwrap();
    let w = g.add_input("w", Shape::new(vec![16, 8]).unwrap(), DType::F32).unwrap();
    let b = g.add_input("b", Shape::new(vec![1, 8]).unwrap(), DType::F32).unwrap();
    let mm = g.add_op(Op::MatMul, vec![x, w]).unwrap();
    let add = g.add_op(Op::Binary(BinaryOp::Add), vec![mm, b]).unwrap();
    let act = g.add_op(Op::Unary(UnaryOp::Relu), vec![add]).unwrap();
    g.mark_output(act).unwrap();

    let cfg = RuntimeConfig {
        opt_level: OptLevel::O3,
        enable_optimization: true,
        enable_profiling: true,
        ..RuntimeConfig::default()
    };
    let session = InferenceSession::new(g, cfg).unwrap();
    assert!(session.compilation_report().is_some());
    assert!(session.memory_plan().is_some());

    let mut inputs = BTreeMap::new();
    inputs.insert("x".to_owned(), Tensor::ones(vec![8, 16], DType::F32).unwrap());
    inputs.insert("w".to_owned(), Tensor::ones(vec![16, 8], DType::F32).unwrap());
    inputs.insert("b".to_owned(), Tensor::ones(vec![1, 8], DType::F32).unwrap());

    let (_outputs, profile_opt) = session.run_named(&inputs).unwrap();
    let profile = profile_opt.expect("profile should be present");

    assert!(!profile.op_profiles.is_empty());
    assert!(profile.total_flops > 0);
    assert!(profile.peak_live_bytes > 0);
    assert!(!profile.slowest_ops(2).is_empty());

    let table = profile.to_summary_table();
    assert!(table.contains("Tarsix Model Profiler Summary"));
    assert!(table.contains("FusedLinear"));

    let chrome = profile.to_chrome_trace_json();
    assert!(chrome.contains("\"traceEvents\""));
    assert!(chrome.contains("\"FusedLinear#"));

    let svg = profile.to_svg_flamegraph();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));
}
