//! Model optimization algorithms, learning-rate scheduling, and gradient clipping.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod clip;
pub mod optimizers;
pub mod scheduler;

pub use clip::{clip_grad_norm, clip_grad_value};
pub use optimizers::{Adam, AdamW, Optimizer, Sgd};
pub use scheduler::{CosineAnnealingLr, LrScheduler, StepLr};
