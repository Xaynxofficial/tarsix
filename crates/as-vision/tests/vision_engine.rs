//! Phase 6 Computer Vision integration tests.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_autograd::Tape;
use as_core::Error;
use as_vision::{
    CnnClassifier, IMAGENET_MEAN, IMAGENET_STD, ImageBuffer, MAX_IMAGE_DIM, VisionTransformer,
    VitConfig, max_pool2d, resize_bilinear, stack_batch, to_normalized_chw_tensor,
};

#[test]
fn image_preprocessing_resize_normalize_and_security_bounds() {
    // Valid 4x4 RGB gradient image
    let mut pixels = Vec::with_capacity(4 * 4 * 3);
    for i in 0..(4 * 4 * 3) {
        pixels.push(((i * 7) % 256) as u8);
    }
    let img = ImageBuffer::new(4, 4, 3, pixels).unwrap();
    let resized = resize_bilinear(&img, 8, 8).unwrap();
    assert_eq!((resized.height, resized.width, resized.channels), (8, 8, 3));

    let tensor = to_normalized_chw_tensor(&resized, &IMAGENET_MEAN, &IMAGENET_STD).unwrap();
    assert_eq!(tensor.dims(), &[1, 3, 8, 8]);

    let batch = stack_batch(&[&img, &resized], 8, 8, &IMAGENET_MEAN, &IMAGENET_STD).unwrap();
    assert_eq!(batch.dims(), &[2, 3, 8, 8]);

    // Security check: oversized dimension rejected
    let err = ImageBuffer::new(MAX_IMAGE_DIM + 1, 1, 3, vec![0; 3]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));

    // Max pool 2x2 stride 2: [2, 3, 8, 8] -> [2, 3, 4, 4]
    let pooled = max_pool2d(&batch, 2, 2).unwrap();
    assert_eq!(pooled.dims(), &[2, 3, 4, 4]);
}

#[test]
fn vision_transformer_and_cnn_forward_and_backward() {
    let img1 = ImageBuffer::new(8, 8, 3, vec![128u8; 8 * 8 * 3]).unwrap();
    let img2 = ImageBuffer::new(8, 8, 3, vec![64u8; 8 * 8 * 3]).unwrap();
    let batch = stack_batch(&[&img1, &img2], 8, 8, &IMAGENET_MEAN, &IMAGENET_STD).unwrap();

    // 1. Vision Transformer (8x8 image, 4x4 patches -> 4 patches)
    let vit_cfg = VitConfig {
        image_size: 8,
        patch_size: 4,
        in_channels: 3,
        embed_dim: 16,
        num_layers: 2,
        num_heads: 4,
        ff_hidden_dim: 32,
        num_classes: 5,
    };
    let vit = VisionTransformer::new(&vit_cfg, 42).unwrap();
    let mut tape = Tape::new();
    let (logits, params) = vit.forward(&mut tape, &batch).unwrap();
    assert_eq!(tape.value(logits).unwrap().dims(), &[2, 5]);
    let loss = tape.mean(logits).unwrap();
    let grads = tape.backward(loss).unwrap();
    for &p in &params {
        assert!(grads.get(p).is_some(), "missing ViT gradient for {p:?}");
    }

    // 2. CNN Classifier
    let cnn = CnnClassifier::new(3, 8, 5, 77).unwrap();
    let mut ctape = Tape::new();
    let x = ctape.var(batch, false);
    let (clogits, cparams) = cnn.forward(&mut ctape, x).unwrap();
    assert_eq!(ctape.value(clogits).unwrap().dims(), &[2, 5]);
    let closs = ctape.mean(clogits).unwrap();
    let cgrads = ctape.backward(closs).unwrap();
    for &p in &cparams {
        assert!(cgrads.get(p).is_some(), "missing CNN gradient for {p:?}");
    }
}
