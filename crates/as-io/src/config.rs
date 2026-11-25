//! Model configuration parsing (`config.json`) and weight compatibility validation.

use as_core::{DType, Error, Result, Shape};

use crate::json::parse_json;
use crate::safetensors::SafeTensors;

/// Parsed external model architecture configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Architecture identifier (e.g. `"llama"`, `"gpt"`, `"vit"`, `"mlp"`).
    pub model_type: String,
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of Transformer / network layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Vocabulary size (0 for non-text models).
    pub vocab_size: usize,
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// Normalization epsilon.
    pub rms_norm_eps: f64,
}

impl ModelConfig {
    /// Parses a [`ModelConfig`] from a `config.json` string.
    pub fn from_json_str(json_str: &str) -> Result<Self> {
        let val = parse_json(json_str)?;
        let obj = val.as_object().ok_or_else(|| Error::InvalidArgument {
            context: "ModelConfig::from_json_str",
            reason: "config.json root must be a JSON object".to_owned(),
        })?;

        let model_type = obj
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_owned();
        let hidden_size = obj
            .get("hidden_size")
            .and_then(|v| v.as_usize())
            .ok_or_else(|| Error::InvalidArgument {
                context: "ModelConfig",
                reason: "missing integer `hidden_size`".to_owned(),
            })?;
        let num_hidden_layers = obj
            .get("num_hidden_layers")
            .and_then(|v| v.as_usize())
            .unwrap_or(1);
        let num_attention_heads = obj
            .get("num_attention_heads")
            .and_then(|v| v.as_usize())
            .unwrap_or(1);
        let vocab_size = obj
            .get("vocab_size")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let intermediate_size = obj
            .get("intermediate_size")
            .and_then(|v| v.as_usize())
            .unwrap_or(hidden_size * 4);
        let rms_norm_eps = obj
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        Ok(Self {
            model_type,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            vocab_size,
            intermediate_size,
            rms_norm_eps,
        })
    }
}

/// Expected tensor specification for validating loaded model weights.
#[derive(Debug, Clone)]
pub struct WeightSpec {
    /// Expected tensor name in the archive.
    pub name: String,
    /// Expected tensor shape.
    pub shape: Shape,
    /// Expected tensor dtype.
    pub dtype: DType,
}

/// Validates that a loaded [`SafeTensors`] archive satisfies every expected [`WeightSpec`].
pub fn validate_weight_compatibility(
    archive: &SafeTensors,
    specs: &[WeightSpec],
) -> Result<()> {
    for spec in specs {
        let tensor = archive
            .tensors
            .get(&spec.name)
            .ok_or_else(|| Error::InvalidArgument {
                context: "validate_weight_compatibility",
                reason: format!("missing required weight `{}`", spec.name),
            })?;
        if tensor.shape() != &spec.shape {
            return Err(Error::ShapeMismatch {
                context: "weight compatibility",
                expected: spec.shape.dims().to_vec(),
                actual: tensor.dims().to_vec(),
            });
        }
        if tensor.dtype() != spec.dtype {
            return Err(Error::DTypeMismatch {
                expected: spec.dtype,
                actual: tensor.dtype(),
            });
        }
    }
    Ok(())
}
