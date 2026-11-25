//! Training and evaluation metrics tracking.

use as_core::{DType, Error, Result};
use as_tensor::Tensor;

/// Summary metrics for a completed training epoch.
#[derive(Debug, Clone, Copy)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Mean training loss across batches.
    pub train_loss: f64,
    /// Optional validation loss.
    pub val_loss: Option<f64>,
    /// Optional validation accuracy in `[0.0, 1.0]`.
    pub val_accuracy: Option<f64>,
}

/// History tracker for loss and evaluation metrics across epochs.
#[derive(Debug, Clone, Default)]
pub struct MetricsTracker {
    history: Vec<EpochMetrics>,
}

impl MetricsTracker {
    /// Creates an empty [`MetricsTracker`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Records metrics for one epoch.
    pub fn record(&mut self, metrics: EpochMetrics) {
        self.history.push(metrics);
    }

    /// Returns recorded epoch metrics.
    pub fn history(&self) -> &[EpochMetrics] {
        &self.history
    }

    /// Returns the most recent epoch metrics, if any.
    pub fn latest(&self) -> Option<&EpochMetrics> {
        self.history.last()
    }
}

/// Computes top-1 classification accuracy for logits `[N, C]` and integer `targets`.
pub fn classification_accuracy(logits: &Tensor, targets: &[usize]) -> Result<f64> {
    if logits.rank() != 2 {
        return Err(Error::InvalidRank {
            context: "classification_accuracy",
            rank: logits.rank(),
            expected: 2,
        });
    }
    let batch = logits.shape().dim(0)?;
    let classes = logits.shape().dim(1)?;
    if targets.len() != batch {
        return Err(Error::BufferLengthMismatch {
            expected: batch,
            actual: targets.len(),
        });
    }
    if batch == 0 || classes == 0 {
        return Ok(0.0);
    }

    let vals = logits.to_dtype(DType::F64)?.to_f64_vec()?;
    let mut correct = 0usize;
    for (row, &target) in targets.iter().enumerate() {
        let row_slice = vals
            .get(row * classes..(row + 1) * classes)
            .ok_or(Error::IndexOutOfBounds {
                index: (row + 1) * classes,
                len: vals.len(),
            })?;
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (c, &v) in row_slice.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
        }
        if best_idx == target {
            correct += 1;
        }
    }
    Ok((correct as f64) / (batch as f64))
}
