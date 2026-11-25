//! Differentiable loss functions for regression and classification.

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, Result};
use as_tensor::Tensor;

use crate::activations::log_softmax;

/// Mean Squared Error loss: `mean((pred - target)^2)`.
pub fn mse_loss(tape: &mut Tape, pred: VarId, target: VarId) -> Result<VarId> {
    let diff = tape.sub(pred, target)?;
    let sq = tape.mul(diff, diff)?;
    tape.mean(sq)
}

/// Cross-Entropy loss for logits of shape `[N, C]` and integer class targets of length `N`.
///
/// Validates all target class indices against `C` to prevent out-of-bounds access
/// on untrusted label data.
pub fn cross_entropy_loss(tape: &mut Tape, logits: VarId, targets: &[usize]) -> Result<VarId> {
    let logits_val = tape.value(logits)?;
    if logits_val.rank() != 2 {
        return Err(Error::InvalidRank {
            context: "cross_entropy_loss",
            rank: logits_val.rank(),
            expected: 2,
        });
    }
    let batch = logits_val.shape().dim(0)?;
    let num_classes = logits_val.shape().dim(1)?;
    let dtype = logits_val.dtype();

    if targets.len() != batch {
        return Err(Error::BufferLengthMismatch {
            expected: batch,
            actual: targets.len(),
        });
    }

    // Build validated one-hot mask of shape [batch, num_classes]
    let mut one_hot = vec![0.0f64; batch * num_classes];
    for (row, &cls) in targets.iter().enumerate() {
        if cls >= num_classes {
            return Err(Error::IndexOutOfBounds {
                index: cls,
                len: num_classes,
            });
        }
        if let Some(slot) = one_hot.get_mut(row * num_classes + cls) {
            *slot = 1.0;
        }
    }

    let mask_f64 = Tensor::from_f64(&one_hot, vec![batch, num_classes])?;
    let mask_t = if dtype == DType::F64 {
        mask_f64
    } else {
        mask_f64.to_dtype(dtype)?
    };
    let mask = tape.var(mask_t, false);

    let log_probs = log_softmax(tape, logits, 1)?;
    let picked = tape.mul(log_probs, mask)?;
    let sum_nll = tape.sum(picked)?;
    let neg_scale = tape.var(Tensor::full(vec![], -1.0 / (batch.max(1) as f64), dtype)?, false);
    tape.mul(sum_nll, neg_scale)
}
