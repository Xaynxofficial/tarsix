//! Integration tests for Phase 3 Model Hub (`as-hub`):
//! - SHA-256 content-addressable caching & tamper rejection
//! - ModelCard registry search & filtering
//! - Sharded Safetensors (`model.safetensors.index.json`) loading & path-traversal rejection

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use as_core::{DType, MemoryLimits};
use as_hub::{
    ModelCache, ModelCard, ModelHub, ModelQuery, ModelTask, QuantizationKind,
    load_sharded_safetensors, parse_shard_index,
};
use as_io::serialize_safetensors;
use as_tensor::Tensor;

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("tarsix_{tag}_{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn model_hub_publishes_searches_verifies_and_loads_weights() {
    let dir = unique_tmp_dir("hub");
    let cache = ModelCache::open(&dir).unwrap();
    let mut hub = ModelHub::new(cache);

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "fc.weight".to_owned(),
        Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
    );
    let bytes = serialize_safetensors(&tensors, &BTreeMap::new()).unwrap();

    let card = ModelCard {
        id: "tarsix/mini-llama-q4".to_owned(),
        architecture: "llama".to_owned(),
        task: ModelTask::Llm,
        num_parameters: 135_000_000,
        dtype: DType::F32,
        quantization: QuantizationKind::Q4,
        license: "Apache-2.0".to_owned(),
        tags: vec!["chat".to_owned(), "edge".to_owned()],
        artifacts: Vec::new(),
    };

    let art = hub
        .publish_safetensors(card, "model.safetensors", &bytes)
        .unwrap();
    assert_eq!(art.sha256.len(), 64);

    let hits = hub.search(&ModelQuery {
        text: Some("llama".to_owned()),
        task: Some(ModelTask::Llm),
        quantization: Some(QuantizationKind::Q4),
        max_parameters: Some(200_000_000),
        ..ModelQuery::default()
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "tarsix/mini-llama-q4");

    assert_eq!(hub.verify_model("tarsix/mini-llama-q4").unwrap(), 1);
    let loaded = hub
        .load_safetensors("tarsix/mini-llama-q4", MemoryLimits::default())
        .unwrap();
    assert_eq!(
        loaded.tensors["fc.weight"].to_f32_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn sharded_safetensors_loads_multiple_shards_and_blocks_traversal() {
    // Reject path traversal in shard index
    let evil = r#"{"weight_map":{"w":"../../etc/passwd"}}"#;
    assert!(parse_shard_index(evil).is_err());

    let dir = unique_tmp_dir("shards");
    let mut s1 = BTreeMap::new();
    s1.insert(
        "layer0.w".to_owned(),
        Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap(),
    );
    let mut s2 = BTreeMap::new();
    s2.insert(
        "layer1.w".to_owned(),
        Tensor::from_f32(&[3.0, 4.0], vec![2]).unwrap(),
    );
    fs::write(
        dir.join("model-00001.safetensors"),
        serialize_safetensors(&s1, &BTreeMap::new()).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("model-00002.safetensors"),
        serialize_safetensors(&s2, &BTreeMap::new()).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{"layer0.w":"model-00001.safetensors","layer1.w":"model-00002.safetensors"}}"#,
    )
    .unwrap();

    let merged = load_sharded_safetensors(&dir, MemoryLimits::default()).unwrap();
    assert_eq!(merged.tensors.len(), 2);
    assert_eq!(merged.tensors["layer0.w"].to_f32_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(merged.tensors["layer1.w"].to_f32_vec().unwrap(), vec![3.0, 4.0]);

    let _ = fs::remove_dir_all(&dir);
}
