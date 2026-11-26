//! Independent Large Language Model (LLM) inference engine.
//!
//! Provides:
//! - Deterministic [`ByteTokenizer`] and [`Tokenizer`] trait
//! - Rotary Positional Embeddings ([`RoPECache`])
//! - Bounded-memory [`KvCache`] for incremental autoregressive decoding
//! - Symmetric Int8 per-channel [`QuantizedLinear`]
//! - Temperature, Top-k, Top-p, and repetition-penalty [`sample_next_token`]
//! - Pre-Norm SwiGLU Transformer Decoder ([`LlmModel`], [`LlmConfig`]) with streaming token generation

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod bpe;
pub mod hf;
pub mod kv_cache;
pub mod model;
pub mod paged_kv;
pub mod quant;
pub mod rope;
pub mod sampler;
pub mod tokenizer;

pub use bpe::BpeTokenizer;
pub use hf::{LoadedHfModel, load_hf_model_dir, parse_hf_llm_config};
pub use kv_cache::KvCache;
pub use model::{LlmConfig, LlmModel};
pub use paged_kv::{PagedCacheStats, PagedKvCache, PagedKvConfig};
pub use quant::{Q4_BLOCK_SIZE, Q4Linear, QuantizedLinear};
pub use rope::RoPECache;
pub use sampler::{SamplingConfig, sample_next_token};
pub use tokenizer::{ByteTokenizer, Tokenizer};
