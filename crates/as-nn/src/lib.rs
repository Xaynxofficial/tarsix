//! Modular neural network framework built on `as-tensor` and `as-autograd`.
//!
//! Provides:
//! - Deterministic PRNG and weight initializers ([`Rng`], [`xavier_uniform`], [`kaiming_normal`])
//! - Trainable [`Param`] and [`Module`] abstractions
//! - Differentiable activations (`sigmoid`, `tanh`, `silu`, `gelu`, `softmax`, `log_softmax`)
//! - Differentiable losses (`mse_loss`, `cross_entropy_loss`)
//! - Core layers: [`Linear`], [`Embedding`], [`LayerNorm`], [`RmsNorm`], [`Dropout`], [`Conv2d`]
//! - Attention & Transformer blocks: [`MultiHeadAttention`], [`TransformerBlock`]

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod activations;
pub mod init;
pub mod layers;
pub mod loss;
pub mod module;
pub mod transformer;

pub use activations::{gelu, log_softmax, sigmoid, silu, softmax, tanh};
pub use init::{Rng, kaiming_normal, xavier_uniform};
pub use layers::{Conv2d, Dropout, Embedding, LayerNorm, Linear, RmsNorm};
pub use loss::{cross_entropy_loss, mse_loss};
pub use module::{Module, Param};
pub use transformer::{MultiHeadAttention, TransformerBlock};
