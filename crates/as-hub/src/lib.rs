//! Zero-dependency, memory-safe Model Hub (`as-hub`).
//!
//! Provides:
//! - Pure safe Rust FIPS 180-4 [`sha256_hex`] cryptographic integrity verification
//! - [`ModelCard`], [`ModelQuery`], and [`ModelHub`] for model discovery, filtering, and lifecycle management
//! - [`ModelCache`] content-addressable local artifact store with path-traversal protection
//! - [`load_sharded_safetensors`] for HuggingFace `model.safetensors.index.json` multi-shard weights

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod cache;
pub mod manifest;
pub mod registry;
pub mod sha256;
pub mod shards;

pub use cache::{CacheStats, ModelCache};
pub use manifest::{ModelArtifact, ModelCard, ModelQuery, ModelTask, QuantizationKind};
pub use registry::ModelHub;
pub use sha256::{sha256, sha256_hex};
pub use shards::{ShardIndex, load_sharded_safetensors, parse_shard_index};
