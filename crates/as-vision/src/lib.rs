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
pub mod detect;
pub mod preprocess;
pub mod vit;

pub use cnn::{CnnClassifier, max_pool2d};
pub use detect::{
    BBox, Detection, LabeledBox, ObjectDetector, decode_detections, detection_loss,
    encode_yolo_targets, iou, nms, render_detections_svg,
};
pub use preprocess::{
    IMAGENET_MEAN, IMAGENET_STD, ImageBuffer, MAX_IMAGE_DIM, resize_bilinear, stack_batch,
    to_normalized_chw_tensor,
};
pub use vit::{PatchEmbed, VisionTransformer, VitConfig};
