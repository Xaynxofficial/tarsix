//! Phase 5 LLM Engine verification tests:
//! - KV-Cache incremental decode vs full-sequence forward numerical equivalence
//! - RoPE L2 norm preservation
//! - Int8 QuantizedLinear accuracy & 4x memory savings
//! - End-to-end streaming token generation with ByteTokenizer

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_llm::{
    BpeTokenizer, ByteTokenizer, LlmConfig, LlmModel, QuantizedLinear, RoPECache, SamplingConfig,
    Tokenizer, load_hf_model_dir,
};
use as_nn::Rng;
use as_tensor::Tensor;

#[test]
fn kv_cache_incremental_decode_matches_full_forward_exactly() {
    let cfg = LlmConfig {
        vocab_size: 32,
        hidden_size: 16,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2, // Llama-3/Qwen Grouped-Query Attention (2:1 Q:KV ratio)
        intermediate_size: 32,
        max_seq_len: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
    };
    let model = LlmModel::new(cfg, 42).unwrap();
    let tokens = [1usize, 5, 9, 3, 12];

    // 1. Full sequence forward without KV-Cache
    let full_logits = model.forward(&tokens, None).unwrap();
    let full_vec = full_logits.to_f32_vec().unwrap();
    let last_full = &full_vec[4 * 32..5 * 32];

    // 2. Prefill first 3 tokens, then incrementally decode token 4 and token 5 using KV-Cache
    let mut cache = model.new_kv_cache(1);
    let _ = model.forward(&tokens[..3], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 3);
    let _ = model.forward(&tokens[3..4], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 4);
    let step5 = model.forward(&tokens[4..5], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 5);
    let last_cached = step5.to_f32_vec().unwrap();

    for (i, (&a, &b)) in last_full.iter().zip(last_cached.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "KV-Cache mismatch at vocab {i}: full={a}, cached={b}"
        );
    }
}

#[test]
fn rope_preserves_vector_l2_norm() {
    let rope = RoPECache::new(64, 8, 10_000.0).unwrap();
    let q = Tensor::from_f32(&[1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -0.5, 2.5], vec![1, 1, 1, 8]).unwrap();
    let q_rot = rope.apply(&q, 17).unwrap();

    let norm_before: f32 = q.to_f32_vec().unwrap().iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_after: f32 = q_rot
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!((norm_before - norm_after).abs() < 1e-5);
}

#[test]
fn int8_quantized_linear_accuracy_and_memory_savings() {
    let w = Tensor::from_f32(
        &[0.5, -0.25, 0.75, -1.0, 0.125, 0.625, -0.5, 0.875],
        vec![2, 4],
    )
    .unwrap();
    let x = Tensor::from_f32(&[1.0, 2.0, -1.0, 0.5], vec![1, 4]).unwrap();

    let fp_out = x.matmul(&w.transpose(0, 1).unwrap()).unwrap().to_f32_vec().unwrap();
    let q_lin = QuantizedLinear::from_f32_weight(&w).unwrap();
    assert_eq!(q_lin.weight_bytes(), 8); // 8 bytes vs 32 bytes f32 (4x reduction)

    let q_out = q_lin.forward(&x).unwrap().to_f32_vec().unwrap();
    for (&a, &b) in fp_out.iter().zip(q_out.iter()) {
        assert!((a - b).abs() < 0.05, "quant error too large: {a} vs {b}");
    }
}

#[test]
fn byte_tokenizer_and_streaming_generation() {
    let tok = ByteTokenizer;
    let prompt = "Hello AI";
    let ids = tok.encode(prompt, true);
    assert_eq!(tok.decode(&ids).unwrap(), prompt);

    let cfg = LlmConfig {
        vocab_size: ByteTokenizer::VOCAB_SIZE,
        hidden_size: 16,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 1,
        intermediate_size: 32,
        max_seq_len: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
    };
    let model = LlmModel::new(cfg, 123).unwrap();
    let mut rng = Rng::from_seed(99);
    let mut streamed = Vec::new();
    let sampling = SamplingConfig {
        temperature: 0.7,
        top_k: 16,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };

    let generated = model
        .generate_stream(&ids, 6, tok.eos_id(), sampling, &mut rng, |t| {
            streamed.push(t);
            true
        })
        .unwrap();
    assert!(!generated.is_empty() && generated.len() <= 6);
    assert_eq!(generated, streamed);
}

#[test]
fn huggingface_bpe_tokenizer_json_encode_and_decode() {
    let tokenizer_json = r#"{
        "added_tokens": [
            {"id": 0, "content": "<|begin_of_text|>"},
            {"id": 1, "content": "<|end_of_text|>"}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 2,
                "h": 3, "e": 4, "l": 5, "o": 6,
                "Ġ": 7, "w": 8, "r": 9, "d": 10,
                "he": 11, "ll": 12, "lo": 13, "hello": 14,
                "Ġw": 15, "or": 16, "ld": 17, "Ġworld": 18
            },
            "merges": [
                "h e",
                "l l",
                "l o",
                "he ll",
                "hell o",
                "Ġ w",
                "o r",
                "l d",
                "Ġw or",
                "Ġwor ld"
            ]
        }
    }"#;

    // Also include intermediate merged tokens in vocab for full BPE chain
    let full_json = tokenizer_json.replace(
        "\"hello\": 14,",
        "\"hell\": 19, \"hello\": 14, \"Ġwor\": 20,",
    );
    let bpe = BpeTokenizer::from_json_str(&full_json).unwrap();
    assert_eq!(bpe.bos_id(), 0);
    assert_eq!(bpe.eos_id(), 1);

    let ids = bpe.encode("hello world", true);
    assert_eq!(ids, vec![0, 14, 18]);
    assert_eq!(bpe.decode(&ids).unwrap(), "hello world");
}

#[test]
fn hf_multi_architecture_qwen2_bf16_tied_embeddings_loader() {
    use std::collections::BTreeMap;
    use std::fs;

    use as_core::{BF16, MemoryLimits};
    use as_io::serialize_safetensors;

    let tmp_dir = std::env::temp_dir().join("tarsix_hf_qwen2_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();

    // 1. Write Qwen2-style config.json (GQA: 4 heads, 2 kv_heads, tied embeddings)
    fs::write(
        tmp_dir.join("config.json"),
        r#"{
            "model_type": "qwen2",
            "vocab_size": 16,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#,
    )
    .unwrap();

    // 2. Write tokenizer.json
    fs::write(
        tmp_dir.join("tokenizer.json"),
        r#"{
            "added_tokens": [
                {"id": 0, "content": "<|begin_of_text|>"},
                {"id": 1, "content": "<|end_of_text|>"}
            ],
            "model": {
                "type": "BPE",
                "vocab": {"<unk>": 2, "h": 3, "i": 4, "hi": 5},
                "merges": ["h i"]
            }
        }"#,
    )
    .unwrap();

    // 3. Write BF16 model.safetensors with Qwen2 Q/K/V biases and NO lm_head.weight (tied!)
    let bf_ones = |n: usize| -> Vec<BF16> { vec![BF16::from_f32(0.05); n] };
    let mut weights = BTreeMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_owned(),
        Tensor::from_bf16(&bf_ones(16 * 16), vec![16, 16]).unwrap(),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.bias".to_owned(),
        Tensor::from_bf16(&bf_ones(16), vec![16]).unwrap(),
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.bias".to_owned(),
        Tensor::from_bf16(&bf_ones(8), vec![8]).unwrap(),
    );
    weights.insert(
        "model.layers.0.self_attn.v_proj.bias".to_owned(),
        Tensor::from_bf16(&bf_ones(8), vec![8]).unwrap(),
    );
    let st_bytes = serialize_safetensors(&weights, &BTreeMap::new()).unwrap();
    fs::write(tmp_dir.join("model.safetensors"), st_bytes).unwrap();

    let loaded = load_hf_model_dir(&tmp_dir, 64, MemoryLimits::default()).unwrap();
    assert_eq!(loaded.model_type, "qwen2");
    assert_eq!(loaded.model.config().num_kv_heads, 2);

    let prompt_ids = loaded.tokenizer.encode("hi", true);
    let mut rng = Rng::from_seed(1);
    let out = loaded
        .model
        .generate_stream(
            &prompt_ids,
            4,
            loaded.tokenizer.eos_id(),
            SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
            &mut rng,
            |_| true,
        )
        .unwrap();
    assert!(!out.is_empty());

    let _ = fs::remove_dir_all(&tmp_dir);
}
