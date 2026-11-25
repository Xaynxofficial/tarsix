//! Independent model execution runtime.
//!
//! Provides:
//! - [`RuntimeConfig`] for device selection, resource limits, optimization, and batch ceilings
//! - [`InferenceSession`] with graph optimization, bound Safetensors weights, and execution profiling
//! - Dynamic request batching ([`InferenceSession::run_batched`])
//! - Streaming step execution ([`InferenceSession::stream_steps`])

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod profiler;
pub mod session;

pub use profiler::{InferenceProfile, RunStats};
pub use session::{InferenceSession, RuntimeConfig};
