//! Independent automatic differentiation engine.
//!
//! Provides:
//! - Reverse-mode automatic differentiation ([`Tape`], [`VarId`], [`Gradients`])
//! - Broadcast-aware VJP rules, gradient accumulation, and real recomputing gradient checkpointing
//! - Forward-mode differentiation via dual tensors ([`DualTensor`])
//! - Central finite-difference numerical gradient validation ([`numerical_gradient`], [`verify_gradients`])

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod forward;
pub mod gradcheck;
pub mod reverse;

pub use forward::DualTensor;
pub use gradcheck::{GradCheckReport, numerical_gradient, verify_gradients};
pub use reverse::{Gradients, Tape, VarId};
