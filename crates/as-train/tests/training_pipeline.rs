//! End-to-end Phase 3 integration tests:
//! - Full XOR MLP training with DataLoader, AdamW, CosineAnnealingLr, grad clipping
//! - Checksum-verified checkpoint save/load + tamper rejection
//! - TransformerBlock, RmsNorm, Embedding, Conv2d, Dropout forward & backward verification

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_autograd::Tape;
use as_core::{DType, Error, MemoryLimits};
use as_nn::{
    Conv2d, Dropout, Embedding, Linear, Module, RmsNorm, Rng, TransformerBlock, cross_entropy_loss,
    gelu,
};
use as_optim::{AdamW, CosineAnnealingLr, LrScheduler, Optimizer, clip_grad_norm};
use as_tensor::Tensor;
use as_train::{
    DataLoader, EpochMetrics, InMemoryDataset, MetricsTracker, Sample, classification_accuracy,
    load_checkpoint, save_checkpoint,
};

struct XorMlp {
    fc1: Linear,
    fc2: Linear,
}

impl XorMlp {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::from_seed(seed);
        Self {
            fc1: Linear::new("fc1", 2, 16, true, DType::F32, &mut rng).unwrap(),
            fc2: Linear::new("fc2", 16, 2, true, DType::F32, &mut rng).unwrap(),
        }
    }

    fn forward(
        &self,
        tape: &mut Tape,
        x: as_autograd::VarId,
    ) -> (as_autograd::VarId, Vec<as_autograd::VarId>) {
        let (h1, p1) = self.fc1.forward(tape, x).unwrap();
        let act = gelu(tape, h1).unwrap();
        let (logits, p2) = self.fc2.forward(tape, act).unwrap();
        let mut all = p1;
        all.extend(p2);
        (logits, all)
    }
}

impl Module for XorMlp {
    fn parameters(&self) -> Vec<&as_nn::Param> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut as_nn::Param> {
        let mut p = self.fc1.parameters_mut();
        p.extend(self.fc2.parameters_mut());
        p
    }
}

#[test]
fn end_to_end_xor_training_convergence_and_safe_checkpointing() {
    let dataset = InMemoryDataset::new(vec![
        Sample {
            features: vec![0.0, 0.0],
            label: 0,
        },
        Sample {
            features: vec![0.0, 1.0],
            label: 1,
        },
        Sample {
            features: vec![1.0, 0.0],
            label: 1,
        },
        Sample {
            features: vec![1.0, 1.0],
            label: 0,
        },
    ])
    .unwrap();

    let loader = DataLoader::new(4, true, false, DType::F32).unwrap();
    let mut model = XorMlp::new(42);
    let mut optim = AdamW::new(0.05, 1e-4);
    let scheduler = CosineAnnealingLr {
        initial_lr: 0.05,
        min_lr: 0.005,
        warmup_steps: 5,
        total_steps: 150,
    };
    let mut rng = Rng::from_seed(1337);
    let mut tracker = MetricsTracker::new();

    for epoch in 0..150 {
        optim.set_learning_rate(scheduler.lr_at(epoch));
        let batches = loader.epoch_batches(&dataset, &mut rng).unwrap();
        let mut epoch_loss = 0.0;

        for batch in &batches {
            let mut tape = Tape::new();
            let x = tape.var(batch.inputs.clone(), false);
            let (logits, param_vars) = model.forward(&mut tape, x);
            let loss = cross_entropy_loss(&mut tape, logits, &batch.labels).unwrap();
            let loss_val = f64::from(tape.value(loss).unwrap().to_f32_vec().unwrap()[0]);
            epoch_loss += loss_val;

            let grads = tape.backward(loss).unwrap();
            let mut grad_tensors: Vec<Tensor> = param_vars
                .iter()
                .map(|&v| grads.get(v).unwrap().clone())
                .collect();
            let _norm = clip_grad_norm(&mut grad_tensors, 5.0).unwrap();

            let grad_refs: Vec<Option<&Tensor>> = grad_tensors.iter().map(Some).collect();
            let mut params_mut = model.parameters_mut();
            optim.step(&mut params_mut, &grad_refs).unwrap();
        }

        tracker.record(EpochMetrics {
            epoch,
            train_loss: epoch_loss / (batches.len() as f64),
            val_loss: None,
            val_accuracy: None,
        });
    }

    let first_loss = tracker.history().first().unwrap().train_loss;
    let final_loss = tracker.latest().unwrap().train_loss;
    assert!(
        final_loss < 0.1 && final_loss < first_loss * 0.2,
        "model failed to converge: first={first_loss}, final={final_loss}"
    );

    // Evaluate accuracy on full XOR dataset
    let eval_x = Tensor::from_f32(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]).unwrap();
    let eval_y = [0usize, 1, 1, 0];
    let mut eval_tape = Tape::new();
    eval_tape.set_grad_enabled(false);
    let vx = eval_tape.var(eval_x.clone(), false);
    let (logits, _) = model.forward(&mut eval_tape, vx);
    let orig_logits = eval_tape.value(logits).unwrap().clone();
    let acc = classification_accuracy(&orig_logits, &eval_y).unwrap();
    assert!((acc - 1.0).abs() < 1e-12, "expected 100% XOR accuracy, got {acc}");

    // Save checkpoint, load into fresh model, verify identical logits
    let ckpt_bytes = save_checkpoint(&model.parameters()).unwrap();
    let mut restored = XorMlp::new(999);
    {
        let mut restored_params = restored.parameters_mut();
        load_checkpoint(&ckpt_bytes, &mut restored_params, MemoryLimits::default()).unwrap();
    }

    let mut eval_tape2 = Tape::new();
    eval_tape2.set_grad_enabled(false);
    let vx2 = eval_tape2.var(eval_x, false);
    let (logits2, _) = restored.forward(&mut eval_tape2, vx2);
    let restored_logits = eval_tape2.value(logits2).unwrap().to_f32_vec().unwrap();
    for (&a, &b) in orig_logits.to_f32_vec().unwrap().iter().zip(restored_logits.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // Tampered checkpoint must be rejected
    let mut tampered = ckpt_bytes;
    let last = tampered.len() - 1;
    tampered[last] ^= 0x80;
    let mut restored_params = restored.parameters_mut();
    let err = load_checkpoint(&tampered, &mut restored_params, MemoryLimits::default()).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));
}

#[test]
fn transformer_embedding_rmsnorm_conv2d_and_dropout_backward() {
    let mut rng = Rng::from_seed(7);

    // 1. Embedding -> TransformerBlock (causal) -> RmsNorm -> Dropout -> sum -> backward
    let emb = Embedding::new("emb", 16, 8, DType::F32, &mut rng).unwrap();
    let block = TransformerBlock::new("blk0", 8, 2, 16, DType::F32, &mut rng).unwrap();
    let rms = RmsNorm::new("rms", 8, 1e-5, DType::F32).unwrap();
    let dropout = Dropout::new(0.2).unwrap();

    let mut tape = Tape::new();
    let (h_emb, p_emb) = emb.forward(&mut tape, &[1, 4, 7, 2, 9, 0], 2, 3).unwrap();
    let (h_blk, p_blk) = block.forward(&mut tape, h_emb, true).unwrap();
    let (h_rms, p_rms) = rms.forward(&mut tape, h_blk).unwrap();
    let h_drop = dropout.forward(&mut tape, h_rms, &mut rng).unwrap();
    let loss = tape.sum(h_drop).unwrap();
    let grads = tape.backward(loss).unwrap();

    for &v in p_emb.iter().chain(p_blk.iter()).chain(p_rms.iter()) {
        assert!(grads.get(v).is_some(), "missing gradient for parameter {v:?}");
    }

    // 2. Conv2d forward + backward shape check: [2, 3, 6, 6] -> k=3, s=1, p=1 -> [2, 4, 6, 6]
    let conv = Conv2d::new("conv1", 3, 4, 3, 3, 1, 1, true, DType::F32, &mut rng).unwrap();
    let mut ctape = Tape::new();
    let img = ctape.var(Tensor::ones(vec![2, 3, 6, 6], DType::F32).unwrap(), false);
    let (feat, p_conv) = conv.forward(&mut ctape, img).unwrap();
    assert_eq!(ctape.value(feat).unwrap().dims(), &[2, 4, 6, 6]);
    let closs = ctape.mean(feat).unwrap();
    let cgrads = ctape.backward(closs).unwrap();
    for &v in &p_conv {
        assert!(cgrads.get(v).is_some());
    }
}
