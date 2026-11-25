//! Phase 4 integration tests: Safetensors security/roundtrip, ModelConfig parsing,
//! InferenceSession weight binding, profiling, dynamic batching, and streaming execution.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;

use as_tensor::{BinaryOp, Tensor, UnaryOp};
use as_core::{BF16, DType, Error, F16, MemoryLimits, Shape};
use as_graph::{Graph, Op};
use as_io::{
    ModelConfig, WeightSpec, deserialize_safetensors, parse_json, serialize_safetensors,
    validate_weight_compatibility,
};
use as_runtime::{InferenceSession, RuntimeConfig};

#[test]
fn safetensors_all_dtypes_roundtrip_and_security_checks() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "w_f32".to_owned(),
        Tensor::from_f32(&[1.5, -2.25, 3.0, 0.5], vec![2, 2]).unwrap(),
    );
    tensors.insert(
        "w_f16".to_owned(),
        Tensor::from_f16(&[F16::from_f32(1.0), F16::from_f32(-0.5)], vec![2]).unwrap(),
    );
    tensors.insert(
        "w_bf16".to_owned(),
        Tensor::from_bf16(&[BF16::from_f32(2.0), BF16::from_f32(4.0)], vec![2]).unwrap(),
    );
    tensors.insert(
        "ids_i64".to_owned(),
        Tensor::from_i64(&[10, 20, 30], vec![3]).unwrap(),
    );

    let mut meta = BTreeMap::new();
    meta.insert("format".to_owned(), "as-engine-v1".to_owned());

    let bytes = serialize_safetensors(&tensors, &meta).unwrap();
    let decoded = deserialize_safetensors(&bytes, MemoryLimits::default()).unwrap();

    assert_eq!(decoded.metadata.get("format").map(String::as_str), Some("as-engine-v1"));
    assert_eq!(
        decoded.tensors["w_f32"].to_f32_vec().unwrap(),
        vec![1.5, -2.25, 3.0, 0.5]
    );
    assert_eq!(
        decoded.tensors["ids_i64"].to_i64_vec().unwrap(),
        vec![10, 20, 30]
    );

    // Weight compatibility validator
    let specs = vec![WeightSpec {
        name: "w_f32".to_owned(),
        shape: Shape::new(vec![2, 2]).unwrap(),
        dtype: DType::F32,
    }];
    validate_weight_compatibility(&decoded, &specs).unwrap();

    // Security check 1: Tight MemoryLimits rejects untrusted file before allocation
    let tight = MemoryLimits::new(8, 8);
    let err = deserialize_safetensors(&bytes, tight).unwrap_err();
    assert!(matches!(err, Error::AllocationLimitExceeded { .. }));

    // Security check 2: Overlapping offsets in crafted Safetensors header must be rejected
    let evil_header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
    let mut evil_bytes = Vec::new();
    evil_bytes.extend_from_slice(&(evil_header.len() as u64).to_le_bytes());
    evil_bytes.extend_from_slice(evil_header.as_bytes());
    evil_bytes.extend_from_slice(&[0u8; 16]);
    let err = deserialize_safetensors(&evil_bytes, MemoryLimits::default()).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));

    // Security check 3: Deeply nested JSON DoS payload rejected
    let deep_json = "[".repeat(64) + &"]".repeat(64);
    assert!(matches!(parse_json(&deep_json), Err(Error::InvalidArgument { .. })));
}

#[test]
fn model_config_and_inference_session_with_safetensors_batching_and_streaming() {
    let cfg = ModelConfig::from_json_str(
        r#"{"model_type":"tiny-llm","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":256,"rms_norm_eps":1e-6}"#,
    )
    .unwrap();
    assert_eq!(cfg.model_type, "tiny-llm");
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.vocab_size, 256);

    // Build graph: y = relu(x @ w + b)
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![1, 2]).unwrap(), DType::F32).unwrap();
    let w = g.add_input("w", Shape::new(vec![2, 2]).unwrap(), DType::F32).unwrap();
    let b = g.add_input("b", Shape::new(vec![1, 2]).unwrap(), DType::F32).unwrap();
    let mm = g.add_op(Op::MatMul, vec![x, w]).unwrap();
    let add = g.add_op(Op::Binary(BinaryOp::Add), vec![mm, b]).unwrap();
    let out = g.add_op(Op::Unary(UnaryOp::Relu), vec![add]).unwrap();
    g.mark_output(out).unwrap();

    // Package weights in Safetensors
    let mut weights = BTreeMap::new();
    weights.insert(
        "w".to_owned(),
        Tensor::from_f32(&[1.0, -1.0, 0.5, 2.0], vec![2, 2]).unwrap(),
    );
    weights.insert("b".to_owned(), Tensor::from_f32(&[0.5, -0.5], vec![1, 2]).unwrap());
    let st_bytes = serialize_safetensors(&weights, &BTreeMap::new()).unwrap();

    let mut session = InferenceSession::new(g, RuntimeConfig::default()).unwrap();
    let bound = session.load_safetensors_bytes(&st_bytes).unwrap();
    assert_eq!(bound, 2);

    // Single profiled run
    let mut req = BTreeMap::new();
    req.insert("x".to_owned(), Tensor::from_f32(&[2.0, 1.0], vec![1, 2]).unwrap());
    let (res, profile) = session.run_named(&req).unwrap();
    assert!(profile.is_some());
    let out_id = session.output_ids()[0];
    // [2, 1] @ [[1, -1], [0.5, 2]] + [0.5, -0.5] = [2.5, 0.0] + [0.5, -0.5] = [3.0, -0.5] -> relu -> [3.0, 0.0]
    assert_eq!(res[&out_id].to_f32_vec().unwrap(), vec![3.0, 0.0]);

    // Dynamic batched run
    let batch_res = session.run_batched(&[req.clone(), req]).unwrap();
    assert_eq!(batch_res.len(), 2);

    // Streaming 3-step run with early stop on step 2
    let mut seen = Vec::new();
    let steps = session
        .stream_steps(
            5,
            |step| {
                let mut m = BTreeMap::new();
                m.insert(
                    "x".to_owned(),
                    Tensor::from_f32(&[(step + 1) as f32, 0.0], vec![1, 2])?,
                );
                Ok(m)
            },
            |step, outputs| {
                seen.push(outputs[&out_id].to_f32_vec()?[0]);
                Ok(step < 2)
            },
        )
        .unwrap();
    assert_eq!(steps, 3);
    assert_eq!(seen, vec![1.5, 2.5, 3.5]);
}
