//! Independent Computer Vision engine.
//!
//! Provides:
//! - Validated image representation, bilinear resizing, CHW normalization, and batching ([`preprocess`])
//! - Patch Embedding and Vision Transformer ([`PatchEmbed`], [`VitConfig`], [`VisionTransformer`])
//! - Trainable Convolutional Image Classifier & Feature Extractor ([`CnnClassifier`])

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod cnn;
pub mod preprocess;
pub mod vit;

pub use cnn::{CnnClassifier, max_pool2d};
pub use preprocess::{
    IMAGENET_MEAN, IMAGENET_STD, ImageBuffer, MAX_IMAGE_DIM, resize_bilinear, stack_batch,
    to_normalized_chw_tensor,
};
pub use vit::{PatchEmbed, VisionTransformer, VitConfig};
