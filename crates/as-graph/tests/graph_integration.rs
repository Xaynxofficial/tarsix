//! Integration tests for `as-graph`: execution, DCE, unary fusion, and safe serialization.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::HashMap;

use as_backend::{BinaryOp, ReduceOp, UnaryOp};
use as_core::{DType, Error, Shape};
use as_graph::{Graph, Op, deserialize_graph, execute_graph, optimize_graph, serialize_graph};
use as_tensor::Tensor;

#[test]
fn graph_dce_fusion_and_execution_equivalence() {
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![2, 3]).unwrap(), DType::F64).unwrap();
    let w = g
        .add_constant(
            Shape::new(vec![3, 2]).unwrap(),
            DType::F64,
            vec![1.0, -1.0, 0.5, 2.0, -0.5, 1.5],
        )
        .unwrap();

    // Dead branch (must be removed by DCE)
    let dead1 = g.add_op(Op::Unary(UnaryOp::Exp), vec![x]).unwrap();
    let _dead2 = g.add_op(Op::Unary(UnaryOp::Neg), vec![dead1]).unwrap();

    // Live path: matmul -> relu -> exp -> sqrt -> sum
    let mm = g.add_op(Op::MatMul, vec![x, w]).unwrap();
    let u1 = g.add_op(Op::Unary(UnaryOp::Relu), vec![mm]).unwrap();
    let u2 = g.add_op(Op::Unary(UnaryOp::Exp), vec![u1]).unwrap();
    let u3 = g.add_op(Op::Unary(UnaryOp::Sqrt), vec![u2]).unwrap();
    let out = g
        .add_op(
            Op::Reduce {
                op: ReduceOp::Sum,
                axis: None,
                keepdim: false,
            },
            vec![u3],
        )
        .unwrap();
    g.mark_output(out).unwrap();

    let (opt, report) = optimize_graph(&g).unwrap();
    assert_eq!(report.dead_nodes_removed, 2);
    assert_eq!(report.unary_ops_fused, 2);
    assert!(opt.nodes().len() < g.nodes().len());

    // Execute both original and optimized graphs and verify bit-level numerical equivalence
    let x_tensor = Tensor::from_f64(&[1.0, 2.0, -1.0, 0.5, -0.5, 3.0], vec![2, 3]).unwrap();

    let mut orig_inputs = HashMap::new();
    orig_inputs.insert(x, x_tensor.clone());
    let orig_res = execute_graph(&g, &orig_inputs).unwrap();
    let orig_val = orig_res[&out].to_f64_vec().unwrap()[0];

    let opt_x = opt
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Input { .. }))
        .unwrap()
        .id;
    let opt_out = opt.outputs()[0];
    let mut opt_inputs = HashMap::new();
    opt_inputs.insert(opt_x, x_tensor);
    let opt_res = execute_graph(&opt, &opt_inputs).unwrap();
    let opt_val = opt_res[&opt_out].to_f64_vec().unwrap()[0];

    assert!((orig_val - opt_val).abs() < 1e-12);
}

#[test]
fn graph_serialization_roundtrip_and_tamper_rejection() {
    let mut g = Graph::new();
    let a = g.add_input("a", Shape::new(vec![4]).unwrap(), DType::F32).unwrap();
    let b = g.add_input("b", Shape::new(vec![4]).unwrap(), DType::F32).unwrap();
    let c = g.add_op(Op::Binary(BinaryOp::Add), vec![a, b]).unwrap();
    g.mark_checkpoint(c).unwrap();
    g.mark_output(c).unwrap();

    let bytes = serialize_graph(&g);
    let restored = deserialize_graph(&bytes).unwrap();
    assert_eq!(g, restored);

    // Flip 1 bit in payload -> must fail integrity checksum
    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    let err = deserialize_graph(&tampered).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));

    // Truncated input -> must fail safely without panicking
    let err = deserialize_graph(&bytes[..6]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));
}
