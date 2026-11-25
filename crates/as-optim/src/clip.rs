//! Gradient norm and value clipping utilities for training stability.

use as_core::{DType, Error, Result};
use as_tensor::Tensor;

/// Clips the global L2 norm of `grads` in place to at most `max_norm`.
///
/// Returns the pre-clipping global L2 norm.
pub fn clip_grad_norm(grads: &mut [Tensor], max_norm: f64) -> Result<f64> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(Error::InvalidArgument {
            context: "clip_grad_norm",
            reason: "max_norm must be finite and positive".to_owned(),
        });
    }
    let mut total_sq = 0.0f64;
    for g in grads.iter() {
        let sq_sum = g
            .to_dtype(DType::F64)?
            .mul(&g.to_dtype(DType::F64)?)?
            .sum()?
            .to_f64_vec()?;
        total_sq += sq_sum.first().copied().unwrap_or(0.0);
    }
    let total_norm = total_sq.sqrt();
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-12);
        for g in grads.iter_mut() {
            let s = Tensor::full(vec![], scale, g.dtype())?;
            *g = g.mul(&s)?;
        }
    }
    Ok(total_norm)
}

/// Clips every element of `grad` into `[-max_abs, max_abs]`.
pub fn clip_grad_value(grad: &Tensor, max_abs: f64) -> Result<Tensor> {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return Err(Error::InvalidArgument {
            context: "clip_grad_value",
            reason: "max_abs must be finite and positive".to_owned(),
        });
    }
    match grad.dtype() {
        DType::F32 => {
            let lim = max_abs as f32;
            let vals: Vec<f32> = grad
                .to_f32_vec()?
                .into_iter()
                .map(|x| x.clamp(-lim, lim))
                .collect();
            Tensor::from_f32(&vals, grad.dims().to_vec())
        }
        DType::F64 => {
            let vals: Vec<f64> = grad
                .to_f64_vec()?
                .into_iter()
                .map(|x| x.clamp(-max_abs, max_abs))
                .collect();
            Tensor::from_f64(&vals, grad.dims().to_vec())
        }
        dtype => Err(Error::UnsupportedDType {
            op: "clip_grad_value",
            dtype,
        }),
    }
}
