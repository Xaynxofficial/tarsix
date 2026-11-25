//! Learning-rate schedules (Step decay and Warmup + Cosine Annealing).

use core::f64::consts::PI;

/// Learning-rate scheduler contract.
pub trait LrScheduler {
    /// Computes the learning rate for `step` (0-indexed).
    fn lr_at(&self, step: usize) -> f64;
}

/// Multiplicative step decay: `initial_lr * gamma^(step / step_size)`.
#[derive(Debug, Clone, Copy)]
pub struct StepLr {
    /// Base learning rate.
    pub initial_lr: f64,
    /// Number of steps between decays.
    pub step_size: usize,
    /// Multiplicative decay factor.
    pub gamma: f64,
}

impl LrScheduler for StepLr {
    fn lr_at(&self, step: usize) -> f64 {
        let periods = (step / self.step_size.max(1)) as i32;
        self.initial_lr * self.gamma.powi(periods)
    }
}

/// Linear warmup followed by cosine annealing decay to `min_lr`.
#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingLr {
    /// Peak learning rate after warmup.
    pub initial_lr: f64,
    /// Minimum learning rate at the end of annealing.
    pub min_lr: f64,
    /// Number of linear warmup steps.
    pub warmup_steps: usize,
    /// Total number of training steps.
    pub total_steps: usize,
}

impl LrScheduler for CosineAnnealingLr {
    fn lr_at(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let progress = ((step + 1) as f64) / (self.warmup_steps as f64);
            return self.initial_lr * progress;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let elapsed = step.saturating_sub(self.warmup_steps).min(decay_steps);
        let ratio = (elapsed as f64) / (decay_steps as f64);
        let cosine = 0.5 * (1.0 + (PI * ratio).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }
}
