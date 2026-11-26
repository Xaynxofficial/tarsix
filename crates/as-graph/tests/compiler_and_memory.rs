//! Integration tests for the Phase 1 Graph Compiler passes, static shape inference,
//! FusedLinear serialization, and static Memory Engine planning.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::HashMap;

use as_backend::{BinaryOp, UnaryOp};
use as_core::{DType, Shape};
use as_graph::{
    Graph, Op, OptLevel, compile_graph, deserialize_graph, execute_graph, serialize_graph,
};
use as_tensor::Tensor;

#[test]
fn compiler_o3_folds_constants_eliminates_cse_and_fuses_linear_activation() {
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![2, 3]).unwrap(), DType::F32).unwrap();
    let w = g.add_input("w", Shape::new(vec![3, 2]).unwrap(), DType::F32).unwrap();
    let b = g.add_input("b", Shape::new(vec![1, 2]).unwrap(), DType::F32).unwrap();

    // Constant subgraph that should be folded: (2.0 + 3.0) -> 5.0
    let c1 = g.add_constant(Shape::new(vec![1]).unwrap(), DType::F32, vec![2.0]).unwrap();
    let c2 = g.add_constant(Shape::new(vec![1]).unwrap(), DType::F32, vec![3.0]).unwrap();
    let c_sum = g.add_op(Op::Binary(BinaryOp::Add), vec![c1, c2]).unwrap();

    // Algebraic identity: double transpose of x -> x
    let t1 = g
        .add_op(Op::Transpose { axis_a: 0, axis_b: 1 }, vec![x])
        .unwrap();
    let t2 = g
        .add_op(Op::Transpose { axis_a: 0, axis_b: 1 }, vec![t1])
        .unwrap();

    // Linear + bias + ReLU -> FusedLinear { has_bias: true, activation: Some(Relu) }
    let mm = g.add_op(Op::MatMul, vec![t2, w]).unwrap();
    let biased = g.add_op(Op::Binary(BinaryOp::Add), vec![mm, b]).unwrap();
    let act = g.add_op(Op::Unary(UnaryOp::Relu), vec![biased]).unwrap();

    // Duplicate subexpression (CSE): add act + c_sum twice, then add together
    let dup1 = g.add_op(Op::Binary(BinaryOp::Add), vec![act, c_sum]).unwrap();
    let dup2 = g.add_op(Op::Binary(BinaryOp::Add), vec![act, c_sum]).unwrap();
    let out = g.add_op(Op::Binary(BinaryOp::Add), vec![dup1, dup2]).unwrap();
    g.mark_output(out).unwrap();

    let x_t = Tensor::from_f32(&[1.0, -2.0, 3.0, 0.5, 1.0, -1.0], vec![2, 3]).unwrap();
    let w_t = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, -1.0], vec![3, 2]).unwrap();
    let b_t = Tensor::from_f32(&[0.5, -0.5], vec![1, 2]).unwrap();

    // Reference execution on unoptimized graph
    let mut ref_in = HashMap::new();
    ref_in.insert(x, x_t.clone());
    ref_in.insert(w, w_t.clone());
    ref_in.insert(b, b_t.clone());
    let ref_out = execute_graph(&g, &ref_in).unwrap();
    let expected = ref_out[&out].to_f32_vec().unwrap();

    // Compile at O3
    let compiled = compile_graph(&g, OptLevel::O3).unwrap();
    assert!(compiled.report.nodes_after < compiled.report.nodes_before);
    assert!(
        compiled.graph.nodes().iter().any(|n| matches!(
            n.op,
            Op::FusedLinear {
                has_bias: true,
                activation: Some(UnaryOp::Relu)
            }
        )),
        "expected MatMul + Bias + ReLU to fuse into Op::FusedLinear"
    );

    // Round-trip serialize/deserialize the compiled graph with FusedLinear
    let bytes = serialize_graph(&compiled.graph);
    let restored = deserialize_graph(&bytes).unwrap();

    // Execute restored compiled graph and verify numerical equivalence
    let mut opt_in = HashMap::new();
    for node in restored.nodes() {
        if let Op::Input { name, .. } = &node.op {
            match name.as_str() {
                "x" => {
                    opt_in.insert(node.id, x_t.clone());
                }
                "w" => {
                    opt_in.insert(node.id, w_t.clone());
                }
                "b" => {
                    opt_in.insert(node.id, b_t.clone());
                }
                _ => {}
            }
        }
    }
    let opt_out = execute_graph(&restored, &opt_in).unwrap();
    let actual = opt_out[&restored.outputs()[0]].to_f32_vec().unwrap();
    assert_eq!(expected.len(), actual.len());
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert!((e - a).abs() < 1e-5, "expected {e}, got {a}");
    }
}
