//! Integration tests for the Phase 2 Hardware Scheduler (`as-scheduler`):
//! - Dynamic host CPU & Apple Silicon Metal GPU discovery
//! - Cost-model op placement (small ops -> CPU, heavy MatMul -> GPU when available)
//! - Parallel DAG wave partitioning & concurrent multi-branch execution equivalence

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::HashMap;

use as_backend::{BinaryOp, UnaryOp};
use as_core::{DType, Device, Shape};
use as_graph::{Graph, Op, execute_graph, infer_graph_metadata};
use as_scheduler::{
    DeviceProfile, DeviceRegistry, HardwareScheduler, PlacementPolicy, build_wave_schedule,
    plan_placement,
};
use as_tensor::Tensor;

#[test]
fn wave_scheduler_executes_parallel_diamond_dag_matching_reference() {
    // Diamond DAG with two independent MatMul+Relu branches merged by Add
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![16, 16]).unwrap(), DType::F32).unwrap();
    let w1 = g.add_input("w1", Shape::new(vec![16, 16]).unwrap(), DType::F32).unwrap();
    let w2 = g.add_input("w2", Shape::new(vec![16, 16]).unwrap(), DType::F32).unwrap();

    let b1 = g.add_op(Op::MatMul, vec![x, w1]).unwrap();
    let b2 = g.add_op(Op::MatMul, vec![x, w2]).unwrap();
    let r1 = g.add_op(Op::Unary(UnaryOp::Relu), vec![b1]).unwrap();
    let r2 = g.add_op(Op::Unary(UnaryOp::Relu), vec![b2]).unwrap();
    let out = g.add_op(Op::Binary(BinaryOp::Add), vec![r1, r2]).unwrap();
    g.mark_output(out).unwrap();

    let wave_sched = build_wave_schedule(&g).unwrap();
    assert!(wave_sched.max_parallelism >= 2);
    assert_eq!(wave_sched.critical_path_depth, 4);

    let x_vals: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.03).sin()).collect();
    let w1_vals: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).cos()).collect();
    let w2_vals: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.07).sin()).collect();

    let mut feed = HashMap::new();
    feed.insert(x, Tensor::from_f32(&x_vals, vec![16, 16]).unwrap());
    feed.insert(w1, Tensor::from_f32(&w1_vals, vec![16, 16]).unwrap());
    feed.insert(w2, Tensor::from_f32(&w2_vals, vec![16, 16]).unwrap());

    let ref_out = execute_graph(&g, &feed).unwrap()[&out].to_f32_vec().unwrap();

    let scheduler = HardwareScheduler::new(PlacementPolicy::default());
    let (sched_out, report) = scheduler.execute(&g, &feed).unwrap();
    let actual = sched_out[&out].to_f32_vec().unwrap();

    assert!(report.parallel_waves >= 2);
    assert_eq!(ref_out.len(), actual.len());
    for (r, a) in ref_out.iter().zip(actual.iter()) {
        assert!((r - a).abs() < 1e-4, "expected {r}, got {a}");
    }
}

#[test]
fn cost_model_places_heavy_matmul_on_gpu_and_tiny_ops_on_cpu() {
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![128, 128]).unwrap(), DType::F32).unwrap();
    let w = g.add_input("w", Shape::new(vec![128, 128]).unwrap(), DType::F32).unwrap();
    let mm = g.add_op(Op::MatMul, vec![x, w]).unwrap();
    let act = g.add_op(Op::Unary(UnaryOp::Relu), vec![mm]).unwrap();
    g.mark_output(act).unwrap();

    let registry = DeviceRegistry::from_profiles(vec![
        DeviceProfile {
            device: Device::Cpu,
            name: "Simulated CPU".to_owned(),
            available: true,
            compute_units: 8,
            peak_gflops_estimate: 64.0,
            memory_bandwidth_gbs: 50.0,
            launch_overhead_us: 1.0,
            transfer_bandwidth_gbs: f64::INFINITY,
        },
        DeviceProfile {
            device: Device::Metal(0),
            name: "Simulated GPU".to_owned(),
            available: true,
            compute_units: 16,
            peak_gflops_estimate: 2_500.0,
            memory_bandwidth_gbs: 200.0,
            launch_overhead_us: 40.0,
            transfer_bandwidth_gbs: 120.0,
        },
    ]);

    let meta = infer_graph_metadata(&g).unwrap();
    let plan = plan_placement(&g, &meta, &registry, PlacementPolicy::default()).unwrap();

    assert_eq!(plan.device_map[&mm], Device::Metal(0));
    assert_eq!(plan.device_map[&act], Device::Cpu);
    assert_eq!(plan.gpu_nodes, 1);
}
