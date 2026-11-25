//! Phase 6 Computer Vision integration tests.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_autograd::Tape;
use as_core::{DType, Error};
use as_nn::Module;
use as_tensor::Tensor;
use as_vision::{
    BBox, CnnClassifier, Detection, IMAGENET_MEAN, IMAGENET_STD, ImageBuffer, LabeledBox,
    MAX_IMAGE_DIM, ObjectDetector, VisionTransformer, VitConfig, decode_detections, detection_loss,
    encode_yolo_targets, iou, max_pool2d, nms, resize_bilinear, stack_batch,
    to_normalized_chw_tensor,
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

#[test]
fn yolo_object_detector_iou_nms_and_end_to_end_training() {
    // 1. Verify IoU and class-aware NMS
    let b1 = BBox { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 };
    let b2 = BBox { cx: 0.52, cy: 0.5, w: 0.4, h: 0.4 };
    let b3 = BBox { cx: 0.1, cy: 0.1, w: 0.2, h: 0.2 };
    assert!(iou(b1, b2) > 0.8);
    assert!(iou(b1, b3) < 0.01);

    let dets = vec![
        Detection { bbox: b1, confidence: 0.95, class_id: 0 },
        Detection { bbox: b2, confidence: 0.80, class_id: 0 }, // suppressed by b1
        Detection { bbox: b2, confidence: 0.85, class_id: 1 }, // different class -> kept
        Detection { bbox: b3, confidence: 0.70, class_id: 0 }, // far away -> kept
    ];
    let kept = nms(&dets, 0.5);
    assert_eq!(kept.len(), 3);

    // 2. End-to-end training of ObjectDetector on a real visual square object at (cx=0.62, cy=0.37)
    let mut detector = ObjectDetector::new(3, 8, 2, 42).unwrap();
    let mut img_data = vec![0.05f32; 3 * 16 * 16];
    for c in 0..3 {
        for y in 4..8 {
            for x in 8..12 {
                img_data[c * 256 + y * 16 + x] = 1.0;
            }
        }
    }
    let img = Tensor::from_f32(&img_data, vec![1, 3, 16, 16]).unwrap();
    let gt_boxes = vec![vec![LabeledBox {
        bbox: BBox { cx: 0.62, cy: 0.37, w: 0.25, h: 0.25 },
        class_id: 1,
    }]];
    let target_grid = encode_yolo_targets(&gt_boxes, 4, 4, 2).unwrap();
    let lr = Tensor::full(vec![], 0.5, DType::F32).unwrap();

    let mut first_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    for step in 0..60 {
        let mut tape = Tape::new();
        let vx = tape.var(img.clone(), false);
        let vt = tape.var(target_grid.clone(), false);
        let (pred, pvars) = detector.forward(&mut tape, vx).unwrap();
        let loss = detection_loss(&mut tape, pred, vt).unwrap();
        let lv = tape.value(loss).unwrap().to_f32_vec().unwrap()[0];
        if step == 0 {
            first_loss = lv;
        }
        final_loss = lv;

        let grads = tape.backward(loss).unwrap();
        let mut params = detector.parameters_mut();
        for (p, &v) in params.iter_mut().zip(pvars.iter()) {
            let g = grads.get(v).unwrap();
            p.set_data(p.data().sub(&g.mul(&lr).unwrap()).unwrap());
        }
    }

    assert!(
        final_loss < first_loss * 0.3,
        "ObjectDetector failed to converge: first={first_loss}, final={final_loss}"
    );

    // 3. Decode predicted detections from the trained model
    let mut eval_tape = Tape::new();
    eval_tape.set_grad_enabled(false);
    let vx = eval_tape.var(img, false);
    let (pred_grid, _) = detector.forward(&mut eval_tape, vx).unwrap();
    let decoded = decode_detections(eval_tape.value(pred_grid).unwrap(), 2, 0.3, 0.5).unwrap();
    assert!(!decoded.is_empty(), "trained detector should output at least 1 detection");
    assert_eq!(decoded[0].class_id, 1);
}
