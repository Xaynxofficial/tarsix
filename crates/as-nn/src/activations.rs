//! Differentiable neural network activation functions on [`Tape`].

use as_autograd::{Tape, VarId};
use as_core::Result;
use as_tensor::Tensor;

/// Differentiable Sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid(tape: &mut Tape, x: VarId) -> Result<VarId> {
    let dtype = tape.value(x)?.dtype();
    let one = tape.var(Tensor::full(vec![], 1.0, dtype)?, false);
    let neg_x = tape.neg(x)?;
    let exp_neg = tape.exp(neg_x)?;
    let denom = tape.add(one, exp_neg)?;
    tape.div(one, denom)
}

/// Differentiable hyperbolic tangent: `2 * sigmoid(2x) - 1`.
pub fn tanh(tape: &mut Tape, x: VarId) -> Result<VarId> {
    let dtype = tape.value(x)?.dtype();
    let two = tape.var(Tensor::full(vec![], 2.0, dtype)?, false);
    let one = tape.var(Tensor::full(vec![], 1.0, dtype)?, false);
    let two_x = tape.mul(two, x)?;
    let sig = sigmoid(tape, two_x)?;
    let two_sig = tape.mul(two, sig)?;
    tape.sub(two_sig, one)
}

/// Differentiable SiLU (Swish): `x * sigmoid(x)`.
pub fn silu(tape: &mut Tape, x: VarId) -> Result<VarId> {
    let s = sigmoid(tape, x)?;
    tape.mul(x, s)
}

/// Differentiable GELU (tanh approximation):
/// `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`.
pub fn gelu(tape: &mut Tape, x: VarId) -> Result<VarId> {
    let dtype = tape.value(x)?.dtype();
    let half = tape.var(Tensor::full(vec![], 0.5, dtype)?, false);
    let one = tape.var(Tensor::full(vec![], 1.0, dtype)?, false);
    let c = tape.var(Tensor::full(vec![], 0.044_715, dtype)?, false);
    let k = tape.var(Tensor::full(vec![], (2.0 / core::f64::consts::PI).sqrt(), dtype)?, false);

    let x2 = tape.mul(x, x)?;
    let x3 = tape.mul(x2, x)?;
    let cx3 = tape.mul(c, x3)?;
    let inner = tape.add(x, cx3)?;
    let scaled = tape.mul(k, inner)?;
    let t = tanh(tape, scaled)?;
    let one_plus_t = tape.add(one, t)?;
    let half_x = tape.mul(half, x)?;
    tape.mul(half_x, one_plus_t)
}

/// Numerically stable differentiable Softmax along `axis`.
pub fn softmax(tape: &mut Tape, x: VarId, axis: usize) -> Result<VarId> {
    // Subtract detached row-wise max for numerical stability without affecting exact gradients
    let max_val = tape.value(x)?.max_axis(axis, true)?;
    let max_const = tape.var(max_val, false);
    let shifted = tape.sub(x, max_const)?;
    let exps = tape.exp(shifted)?;
    let sum_exps = tape.sum_axis(exps, axis, true)?;
    tape.div(exps, sum_exps)
}

/// Numerically stable differentiable Log-Softmax along `axis`.
pub fn log_softmax(tape: &mut Tape, x: VarId, axis: usize) -> Result<VarId> {
    let max_val = tape.value(x)?.max_axis(axis, true)?;
    let max_const = tape.var(max_val, false);
    let shifted = tape.sub(x, max_const)?;
    let exps = tape.exp(shifted)?;
    let sum_exps = tape.sum_axis(exps, axis, true)?;
    let log_sum = tape.ln(sum_exps)?;
    tape.sub(shifted, log_sum)
}
