//! Safe, zero-dependency model interoperability (`safetensors` and `config.json`).
//!
//! All external model files and headers are parsed as untrusted input with strict
//! recursion depth, header size, offset-overlap, and [`as_core::MemoryLimits`] checks.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod config;
pub mod json;
pub mod safetensors;

pub use config::{ModelConfig, WeightSpec, validate_weight_compatibility};
pub use json::{JsonValue, parse_json};
pub use safetensors::{
    MAX_SAFETENSORS_HEADER_BYTES, SafeTensors, TensorInfo, deserialize_safetensors,
    serialize_safetensors,
};
