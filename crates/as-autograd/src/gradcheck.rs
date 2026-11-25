//! Central finite-difference numerical gradient validation (`gradcheck`).

use as_core::{DType, Error, Result};
use as_tensor::Tensor;

/// Diagnostic report returned by [`verify_gradients`].
#[derive(Debug, Clone)]
pub struct GradCheckReport {
    /// Maximum absolute difference observed across all elements.
    pub max_abs_diff: f64,
    /// Maximum relative difference observed across all elements.
    pub max_rel_diff: f64,
    /// Whether all elements satisfied `|a - n| <= atol + rtol * |n|`.
    pub passed: bool,
}

/// Computes the numerical gradient of scalar-valued function `f` at `input` (in `F64`)
/// using central finite differences: `(f(x + eps) - f(x - eps)) / (2 * eps)`.
pub fn numerical_gradient<F>(input: &Tensor, eps: f64, f: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<f64>,
{
    if input.dtype() != DType::F64 {
        return Err(Error::DTypeMismatch {
            expected: DType::F64,
            actual: input.dtype(),
        });
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::InvalidArgument {
            context: "numerical_gradient",
            reason: "eps must be finite and positive".to_owned(),
        });
    }

    let base = input.to_f64_vec()?;
    let mut grad = vec![0.0f64; base.len()];
    let mut perturbed = base.clone();

    for i in 0..base.len() {
        let orig = base.get(i).copied().unwrap_or(0.0);

        if let Some(slot) = perturbed.get_mut(i) {
            *slot = orig + eps;
        }
        let t_plus = Tensor::from_f64(&perturbed, input.dims().to_vec())?;
        let f_plus = f(&t_plus)?;

        if let Some(slot) = perturbed.get_mut(i) {
            *slot = orig - eps;
        }
        let t_minus = Tensor::from_f64(&perturbed, input.dims().to_vec())?;
        let f_minus = f(&t_minus)?;

        if let Some(slot) = perturbed.get_mut(i) {
            *slot = orig;
        }
        if let Some(g_slot) = grad.get_mut(i) {
            *g_slot = (f_plus - f_minus) / (2.0 * eps);
        }
    }

    Tensor::from_f64(&grad, input.dims().to_vec())
}

/// Compares an analytical gradient against a numerical gradient within `(atol, rtol)`.
pub fn verify_gradients(
    analytical: &Tensor,
    numerical: &Tensor,
    atol: f64,
    rtol: f64,
) -> Result<GradCheckReport> {
    if analytical.shape() != numerical.shape() {
        return Err(Error::ShapeMismatch {
            context: "verify_gradients",
            expected: numerical.dims().to_vec(),
            actual: analytical.dims().to_vec(),
        });
    }
    let a = analytical.to_dtype(DType::F64)?.to_f64_vec()?;
    let n = numerical.to_dtype(DType::F64)?.to_f64_vec()?;

    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    let mut passed = true;

    for (&av, &nv) in a.iter().zip(n.iter()) {
        let abs_diff = (av - nv).abs();
        let rel_diff = abs_diff / nv.abs().max(1e-12);
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
        if abs_diff > atol + rtol * nv.abs() || av.is_nan() || nv.is_nan() {
            passed = false;
        }
    }

    Ok(GradCheckReport {
        max_abs_diff,
        max_rel_diff,
        passed,
    })
}
