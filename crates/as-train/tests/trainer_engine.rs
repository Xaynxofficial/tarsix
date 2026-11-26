//! Integration tests for the Phase 2 Declarative Training Pipeline (`Trainer`,
//! gradient accumulation, callbacks, EarlyStopping, and checksummed `TrainState` resume).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_autograd::{Tape, VarId};
use as_core::{DType, MemoryLimits};
use as_nn::{Linear, Module, Param, Rng, cross_entropy_loss, gelu};
use as_optim::{AdamW, CosineAnnealingLr};
use as_train::{
    Batch, BestCheckpointCallback, EarlyStopping, InMemoryDataset, JsonlLoggerCallback, Sample,
    TrainConfig, TrainState, Trainer, classification_accuracy,
};

struct TinyMlp {
    fc1: Linear,
    fc2: Linear,
}

impl TinyMlp {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::from_seed(seed);
        Self {
            fc1: Linear::new("fc1", 2, 12, true, DType::F32, &mut rng).unwrap(),
            fc2: Linear::new("fc2", 12, 2, true, DType::F32, &mut rng).unwrap(),
        }
    }

    fn forward(&self, tape: &mut Tape, x: VarId) -> (VarId, Vec<VarId>) {
        let (h, mut p) = self.fc1.forward(tape, x).unwrap();
        let act = gelu(tape, h).unwrap();
        let (logits, p2) = self.fc2.forward(tape, act).unwrap();
        p.extend(p2);
        (logits, p)
    }
}

impl Module for TinyMlp {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = self.fc1.parameters_mut();
        p.extend(self.fc2.parameters_mut());
        p
    }
}

#[test]
fn trainer_converges_with_grad_accum_callbacks_and_resumes_from_state() {
    let dataset = InMemoryDataset::new(vec![
        Sample { features: vec![0.0, 0.0], label: 0 },
        Sample { features: vec![0.0, 1.0], label: 1 },
        Sample { features: vec![1.0, 0.0], label: 1 },
        Sample { features: vec![1.0, 1.0], label: 0 },
    ])
    .unwrap();

    let mut model = TinyMlp::new(7);
    let mut opt = AdamW::new(0.05, 0.0);
    let sched = CosineAnnealingLr {
        initial_lr: 0.05,
        min_lr: 0.005,
        warmup_steps: 2,
        total_steps: 40,
    };

    let cfg = TrainConfig {
        epochs: 30,
        batch_size: 2,
        grad_accum_steps: 2, // accumulates 2 microbatches of size 2 = effective batch 4
        max_grad_norm: Some(2.0),
        shuffle: false,
        seed: 42,
        dtype: DType::F32,
        eval_every: 1,
        loss_scale: 1.0,
    };

    let mut trainer = Trainer::new(cfg).unwrap();
    trainer
        .add_callback(Box::new(EarlyStopping::new(25, 1e-6)))
        .add_callback(Box::new(BestCheckpointCallback::new()))
        .add_callback(Box::new(JsonlLoggerCallback::new()));

    let step_fn = |m: &TinyMlp, tape: &mut Tape, batch: &Batch| {
        let x = tape.var(batch.inputs.clone(), false);
        let (logits, params) = m.forward(tape, x);
        let loss = cross_entropy_loss(tape, logits, &batch.labels)?;
        Ok((loss, params))
    };

    let eval_fn = |m: &TinyMlp, batch: &Batch| {
        let mut tape = Tape::new();
        let x = tape.var(batch.inputs.clone(), false);
        let (logits, _) = m.forward(&mut tape, x);
        let loss = cross_entropy_loss(&mut tape, logits, &batch.labels)?;
        let loss_val = tape.value(loss)?.to_dtype(DType::F64)?.to_f64_vec()?[0];
        let acc = classification_accuracy(tape.value(logits)?, &batch.labels)?;
        Ok((loss_val, Some(acc)))
    };

    let report = trainer
        .fit(
            &mut model,
            &mut opt,
            &dataset,
            Some(&dataset),
            Some(&sched),
            step_fn,
            Some(eval_fn),
        )
        .unwrap();

    assert!(!report.history.is_empty());
    assert!(report.total_steps >= 10);
    assert!(report.best_checkpoint.is_some());
    let first_loss = report.history.first().unwrap().train_loss;
    let last_loss = report.history.last().unwrap().train_loss;
    assert!(
        last_loss < first_loss,
        "expected loss to decrease: first={first_loss}, last={last_loss}"
    );

    // Verify TrainState round-trip serialization + tamper rejection + resume
    let state = trainer.snapshot_state(&model, report.history.len() - 1).unwrap();
    let bytes = state.to_bytes();
    let restored = TrainState::from_bytes(&bytes).unwrap();
    assert_eq!(restored.next_epoch, report.history.len());
    assert_eq!(restored.global_step, report.total_steps);

    let mut tampered = bytes;
    tampered[20] ^= 0xFF;
    assert!(TrainState::from_bytes(&tampered).is_err());

    let mut fresh_model = TinyMlp::new(999);
    let mut resumed_trainer = Trainer::new(TrainConfig::default()).unwrap();
    resumed_trainer
        .resume_from(&restored, &mut fresh_model, MemoryLimits::default())
        .unwrap();
}
