//! Universal HuggingFace Safetensors directory loader supporting Llama-2/3,
//! SmolLM/SmolLM2, TinyLlama, Mistral, Qwen2/Qwen2.5, and compatible decoder families.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use as_core::{Error, MemoryLimits, Result};
use as_io::{SafeTensors, deserialize_safetensors, parse_json};

use crate::bpe::BpeTokenizer;
use crate::model::{LlmConfig, LlmModel};

/// Metadata + loaded model + BPE tokenizer from a HuggingFace model directory.
#[derive(Debug)]
pub struct LoadedHfModel {
    /// Architecture family reported in `config.json` (e.g. `"llama"`, `"qwen2"`, `"mistral"`).
    pub model_type: String,
    /// Loaded Transformer decoder model.
    pub model: LlmModel,
    /// Loaded BPE tokenizer.
    pub tokenizer: BpeTokenizer,
}

/// Loads any compatible HuggingFace Safetensors LLM directory containing:
/// - `config.json`
/// - `tokenizer.json`
/// - `model.safetensors` OR sharded `model.safetensors.index.json`
pub fn load_hf_model_dir(
    dir: impl AsRef<Path>,
    max_seq_len_cap: usize,
    limits: MemoryLimits,
) -> Result<LoadedHfModel> {
    let dir = dir.as_ref();

    // 1. Parse config.json
    let cfg_path = dir.join("config.json");
    let cfg_str = fs::read_to_string(&cfg_path).map_err(|e| Error::InvalidArgument {
        context: "load_hf_model_dir",
        reason: format!("failed to read `{}`: {e}", cfg_path.display()),
    })?;
    let (model_type, llm_cfg) = parse_hf_llm_config(&cfg_str, max_seq_len_cap)?;

    // 2. Parse tokenizer.json
    let tok_path = dir.join("tokenizer.json");
    let tok_str = fs::read_to_string(&tok_path).map_err(|e| Error::InvalidArgument {
        context: "load_hf_model_dir",
        reason: format!("failed to read `{}`: {e}", tok_path.display()),
    })?;
    let tokenizer = BpeTokenizer::from_json_str(&tok_str)?;

    // 3. Load single or sharded Safetensors weights
    let archive = load_safetensors_dir(dir, limits)?;

    // 4. Construct model and bind weights
    let mut model = LlmModel::new(llm_cfg, 0)?;
    model.load_from_safetensors(&archive)?;

    Ok(LoadedHfModel {
        model_type,
        model,
        tokenizer,
    })
}

/// Parses a HuggingFace `config.json` into `(model_type, LlmConfig)`.
pub fn parse_hf_llm_config(json_str: &str, max_seq_len_cap: usize) -> Result<(String, LlmConfig)> {
    let root = parse_json(json_str)?;
    let obj = root.as_object().ok_or_else(|| Error::InvalidArgument {
        context: "parse_hf_llm_config",
        reason: "config.json root must be a JSON object".to_owned(),
    })?;

    let model_type = obj
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_owned();

    let vocab_size = req_usize(obj, "vocab_size")?;
    let hidden_size = req_usize(obj, "hidden_size")?;
    let num_layers = req_usize(obj, "num_hidden_layers")?;
    let num_heads = req_usize(obj, "num_attention_heads")?;
    let num_kv_heads = obj
        .get("num_key_value_heads")
        .and_then(|v| v.as_usize())
        .unwrap_or(num_heads);
    let intermediate_size = obj
        .get("intermediate_size")
        .and_then(|v| v.as_usize())
        .unwrap_or(hidden_size * 4);
    let max_pos = obj
        .get("max_position_embeddings")
        .and_then(|v| v.as_usize())
        .unwrap_or(2048);
    let max_seq_len = max_pos.min(max_seq_len_cap.max(1));
    let rms_norm_eps = obj
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5);
    let rope_theta = obj
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0);

    Ok((
        model_type,
        LlmConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
        },
    ))
}

fn req_usize(obj: &BTreeMap<String, as_io::JsonValue>, key: &str) -> Result<usize> {
    obj.get(key)
        .and_then(|v| v.as_usize())
        .ok_or_else(|| Error::InvalidArgument {
            context: "parse_hf_llm_config",
            reason: format!("missing integer `{key}` in config.json"),
        })
}

fn load_safetensors_dir(dir: &Path, limits: MemoryLimits) -> Result<SafeTensors> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        let bytes = fs::read(&single).map_err(|e| Error::InvalidArgument {
            context: "load_safetensors_dir",
            reason: format!("failed to read `{}`: {e}", single.display()),
        })?;
        return deserialize_safetensors(&bytes, limits);
    }

    // Sharded Safetensors: model.safetensors.index.json
    let index_path = dir.join("model.safetensors.index.json");
    let idx_str = fs::read_to_string(&index_path).map_err(|e| Error::InvalidArgument {
        context: "load_safetensors_dir",
        reason: format!(
            "neither `model.safetensors` nor `model.safetensors.index.json` found in `{}`: {e}",
            dir.display()
        ),
    })?;
    let idx_json = parse_json(&idx_str)?;
    let weight_map = idx_json
        .as_object()
        .and_then(|o| o.get("weight_map"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| Error::InvalidArgument {
            context: "load_safetensors_dir",
            reason: "invalid `weight_map` in model.safetensors.index.json".to_owned(),
        })?;

    let mut shard_files = BTreeSet::new();
    for val in weight_map.values() {
        if let Some(fname) = val.as_str() {
            // Security check: reject path traversal in untrusted shard filenames
            if fname.contains("..") || fname.contains('/') || fname.contains('\\') {
                return Err(Error::InvalidArgument {
                    context: "load_safetensors_dir",
                    reason: format!("unsafe shard filename `{fname}`"),
                });
            }
            shard_files.insert(fname.to_owned());
        }
    }

    let mut merged_tensors = BTreeMap::new();
    let mut merged_meta = BTreeMap::new();
    for shard in shard_files {
        let shard_path = dir.join(&shard);
        let bytes = fs::read(&shard_path).map_err(|e| Error::InvalidArgument {
            context: "load_safetensors_dir",
            reason: format!("failed to read shard `{}`: {e}", shard_path.display()),
        })?;
        let part = deserialize_safetensors(&bytes, limits)?;
        merged_tensors.extend(part.tensors);
        merged_meta.extend(part.metadata);
    }

    Ok(SafeTensors {
        tensors: merged_tensors,
        metadata: merged_meta,
    })
}
