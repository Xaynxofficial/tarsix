//! Safe multi-shard Safetensors loader (`model.safetensors.index.json`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use as_core::{Error, MemoryLimits, Result};
use as_io::{JsonValue, SafeTensors, deserialize_safetensors, parse_json};

/// Parsed HuggingFace `model.safetensors.index.json` shard mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardIndex {
    /// Mapping from tensor name -> relative shard filename.
    pub weight_map: BTreeMap<String, String>,
    /// Deduplicated ordered list of shard filenames.
    pub shard_files: Vec<String>,
}

/// Safely parses a `model.safetensors.index.json` payload and validates all shard filenames
/// against path traversal (`..`, `/`, `\`).
pub fn parse_shard_index(json_str: &str) -> Result<ShardIndex> {
    let val = parse_json(json_str)?;
    let wm_obj = val
        .get("weight_map")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| Error::InvalidArgument {
            context: "parse_shard_index",
            reason: "missing `weight_map` object in shard index JSON".to_owned(),
        })?;

    let mut weight_map = BTreeMap::new();
    let mut unique_shards = BTreeSet::new();

    for (tensor_name, shard_val) in wm_obj {
        let shard_name = shard_val.as_str().ok_or_else(|| Error::InvalidArgument {
            context: "parse_shard_index",
            reason: format!("non-string shard filename for tensor `{tensor_name}`"),
        })?;
        validate_safe_filename(shard_name)?;
        weight_map.insert(tensor_name.clone(), shard_name.to_owned());
        unique_shards.insert(shard_name.to_owned());
    }

    Ok(ShardIndex {
        weight_map,
        shard_files: unique_shards.into_iter().collect(),
    })
}

/// Loads a sharded Safetensors model directory (`model.safetensors.index.json` + shard files)
/// into a single unified [`SafeTensors`] archive, enforcing `limits` and path-traversal security.
pub fn load_sharded_safetensors(dir: &Path, limits: MemoryLimits) -> Result<SafeTensors> {
    let index_path = dir.join("model.safetensors.index.json");
    let index_str = fs::read_to_string(&index_path).map_err(|e| Error::InvalidArgument {
        context: "load_sharded_safetensors",
        reason: format!("failed to read {}: {e}", index_path.display()),
    })?;
    let index = parse_shard_index(&index_str)?;

    let mut merged = BTreeMap::new();
    let mut metadata = BTreeMap::new();

    for shard_file in &index.shard_files {
        let shard_path = dir.join(shard_file);
        let bytes = fs::read(&shard_path).map_err(|e| Error::InvalidArgument {
            context: "load_sharded_safetensors",
            reason: format!("failed to read shard {}: {e}", shard_path.display()),
        })?;
        let archive = deserialize_safetensors(&bytes, limits)?;
        for (k, v) in archive.metadata {
            metadata.insert(k, v);
        }
        for (name, tensor) in archive.tensors {
            merged.insert(name, tensor);
        }
    }

    // Verify all tensors declared in weight_map were actually present
    for expected_name in index.weight_map.keys() {
        if !merged.contains_key(expected_name) {
            return Err(Error::InvalidArgument {
                context: "load_sharded_safetensors",
                reason: format!("tensor `{expected_name}` declared in index missing from shards"),
            });
        }
    }

    Ok(SafeTensors {
        metadata,
        tensors: merged,
    })
}

fn validate_safe_filename(name: &str) -> Result<()> {
    if name.is_empty()
        || name.contains("..")
        || name.contains('/')
        || name.contains('\\')
        || name.starts_with('.')
    {
        return Err(Error::InvalidArgument {
            context: "validate_safe_filename",
            reason: format!("unsafe shard filename `{name}` rejected"),
        });
    }
    Ok(())
}
