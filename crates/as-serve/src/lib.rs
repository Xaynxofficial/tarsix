//! Zero-dependency, memory-safe Inference Server (`as-serve`).
//!
//! Provides:
//! - Hardened HTTP/1.1 parser and Server-Sent Events (SSE) streaming ([`http`])
//! - Real concurrent [`DynamicBatcher`] grouping requests by `max_batch_size` and `max_wait_ms`
//! - [`InferenceServer`] with `/health`, `/metrics` (Prometheus), `/v1/models`, `/v1/infer`,
//!   and OpenAI-compatible `/v1/chat/completions` (JSON & SSE streaming)

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod batcher;
pub mod continuous;
pub mod http;
pub mod server;

pub use batcher::{BatchConfig, BatcherStats, DynamicBatcher};
pub use continuous::{
    ActiveSequence, ContinuousBatchConfig, ContinuousBatchStats, ContinuousBatcher,
    SequenceRequest, StepGenerator, TokenStepOutput,
};
pub use http::{HttpRequest, HttpResponse};
pub use server::{InferenceServer, ServerConfig, ServerHandle, ServerMetrics};
