//! Reverse-mode automatic differentiation (backpropagation), gradient accumulation,
//! and just-in-time recomputing gradient checkpointing.

use std::collections::HashMap;
use std::rc::Rc;

use as_core::{DType, Error, Result, Shape};
use as_tensor::Tensor;

/// Strongly typed handle to a recorded variable on a [`Tape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub usize);

/// Type alias for a checkpointed subgraph recomputation closure.
type RecomputeFn = Rc<dyn Fn(&mut Tape, &[VarId]) -> Result<VarId>>;

/// Gradient accumulator mapping [`VarId`] to accumulated gradient [`Tensor`]s.
#[derive(Debug, Clone, Default)]
pub struct Gradients {
    grads: HashMap<VarId, Tensor>,
}

impl Gradients {
    /// Creates an empty gradient map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the accumulated gradient for `var`, if any.
    pub fn get(&self, var: VarId) -> Option<&Tensor> {
        self.grads.get(&var)
    }

    /// Accumulates `grad` into `var`'s gradient slot (adding if already present).
    pub fn accumulate(&mut self, var: VarId, grad: Tensor) -> Result<()> {
        if let Some(existing) = self.grads.get_mut(&var) {
            *existing = existing.add(&grad)?;
        } else {
            self.grads.insert(var, grad);
        }
        Ok(())
    }

    /// Clears all accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }

    /// Returns the number of variables with stored gradients.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Returns `true` if no gradients are stored.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

#[derive(Clone)]
enum TapeOp {
    Leaf,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Relu,
    Powf(f64),
    MatMul,
    Reshape {
        input_shape: Shape,
    },
    Transpose {
        axis_a: usize,
        axis_b: usize,
    },
    Sum {
        input_shape: Shape,
        axis: Option<usize>,
        keepdim: bool,
    },
    Mean {
        input_shape: Shape,
        axis: Option<usize>,
        keepdim: bool,
        count: usize,
    },
    Im2Col {
        input_shape: Shape,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    },
    Checkpoint {
        recompute: RecomputeFn,
    },
}

struct TapeEntry {
    op: TapeOp,
    inputs: Vec<VarId>,
    value: Tensor,
    requires_grad: bool,
}

/// Dynamic reverse-mode automatic differentiation tape.
///
/// Records tensor operations in forward order and computes exact gradients
/// via reverse-mode backpropagation.
#[derive(Default)]
pub struct Tape {
    entries: Vec<TapeEntry>,
    grad_enabled: bool,
}

impl Tape {
    /// Creates a new [`Tape`] with gradient recording enabled.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            grad_enabled: true,
        }
    }

    /// Enables or disables gradient recording (analogous to `no_grad` mode).
    pub fn set_grad_enabled(&mut self, enabled: bool) {
        self.grad_enabled = enabled;
    }

    /// Returns whether gradient recording is currently enabled.
    pub const fn is_grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// Registers a leaf tensor on the tape.
    pub fn var(&mut self, tensor: Tensor, requires_grad: bool) -> VarId {
        let id = VarId(self.entries.len());
        self.entries.push(TapeEntry {
            op: TapeOp::Leaf,
            inputs: Vec::new(),
            value: tensor,
            requires_grad: self.grad_enabled && requires_grad,
        });
        id
    }

    /// Borrows the forward primal [`Tensor`] for `var`.
    pub fn value(&self, var: VarId) -> Result<&Tensor> {
        self.entries
            .get(var.0)
            .map(|e| &e.value)
            .ok_or(Error::IndexOutOfBounds {
                index: var.0,
                len: self.entries.len(),
            })
    }

    /// Returns whether `var` requires gradient computation.
    pub fn requires_grad(&self, var: VarId) -> Result<bool> {
        self.entries
            .get(var.0)
            .map(|e| e.requires_grad)
            .ok_or(Error::IndexOutOfBounds {
                index: var.0,
                len: self.entries.len(),
            })
    }

    fn record(&mut self, op: TapeOp, inputs: Vec<VarId>, value: Tensor) -> Result<VarId> {
        let mut req = false;
        if self.grad_enabled {
            for &inp in &inputs {
                if self.requires_grad(inp)? {
                    req = true;
                    break;
                }
            }
        }
        let id = VarId(self.entries.len());
        self.entries.push(TapeEntry {
            op,
            inputs,
            value,
            requires_grad: req,
        });
        Ok(id)
    }

    // --- Differentiable Operations ---

    /// Differentiable element-wise addition with broadcasting.
    pub fn add(&mut self, lhs: VarId, rhs: VarId) -> Result<VarId> {
        let out = self.value(lhs)?.add(self.value(rhs)?)?;
        self.record(TapeOp::Add, vec![lhs, rhs], out)
    }

    /// Differentiable element-wise subtraction with broadcasting.
    pub fn sub(&mut self, lhs: VarId, rhs: VarId) -> Result<VarId> {
        let out = self.value(lhs)?.sub(self.value(rhs)?)?;
        self.record(TapeOp::Sub, vec![lhs, rhs], out)
    }

    /// Differentiable element-wise multiplication with broadcasting.
    pub fn mul(&mut self, lhs: VarId, rhs: VarId) -> Result<VarId> {
        let out = self.value(lhs)?.mul(self.value(rhs)?)?;
        self.record(TapeOp::Mul, vec![lhs, rhs], out)
    }

    /// Differentiable element-wise division with broadcasting.
    pub fn div(&mut self, lhs: VarId, rhs: VarId) -> Result<VarId> {
        let out = self.value(lhs)?.div(self.value(rhs)?)?;
        self.record(TapeOp::Div, vec![lhs, rhs], out)
    }

    /// Differentiable negation (`-x`).
    pub fn neg(&mut self, x: VarId) -> Result<VarId> {
        let out = self.value(x)?.neg()?;
        self.record(TapeOp::Neg, vec![x], out)
    }

    /// Differentiable absolute value (`|x|`; subgradient 0 at 0).
    pub fn abs(&mut self, x: VarId) -> Result<VarId> {
        let out = self.value(x)?.abs()?;
        self.record(TapeOp::Abs, vec![x], out)
    }

    /// Differentiable exponential (`e^x`).
    pub fn exp(&mut self, x: VarId) -> Result<VarId> {
        let out = self.value(x)?.exp()?;
        self.record(TapeOp::Exp, vec![x], out)
    }

    /// Differentiable natural logarithm (`ln(x)`).
    pub fn ln(&mut self, x: VarId) -> Result<VarId> {
        let out = self.value(x)?.ln()?;
        self.record(TapeOp::Ln, vec![x], out)
    }

    /// Differentiable square root (`sqrt(x)`).
    pub fn sqrt(&mut self, x: VarId) -> Result<VarId> {
        let out = self.value(x)?.sqrt()?;
        self.record(TapeOp::Sqrt, vec![x], out)
    }

    /// Differentiable ReLU (`max(0, x)`).
    pub fn relu(&mut self, x: VarId) -> Result<VarId> {
        let out = self.value(x)?.relu()?;
        self.record(TapeOp::Relu, vec![x], out)
    }

    /// Differentiable element-wise power (`x^p`).
    pub fn powf(&mut self, x: VarId, p: f64) -> Result<VarId> {
        let out = self.value(x)?.powf(p)?;
        self.record(TapeOp::Powf(p), vec![x], out)
    }

    /// Differentiable 2D or batched N-D matrix multiplication.
    pub fn matmul(&mut self, lhs: VarId, rhs: VarId) -> Result<VarId> {
        let out = self.value(lhs)?.matmul(self.value(rhs)?)?;
        self.record(TapeOp::MatMul, vec![lhs, rhs], out)
    }

    /// Differentiable reshape.
    pub fn reshape(&mut self, x: VarId, new_shape: impl Into<Vec<usize>>) -> Result<VarId> {
        let input_shape = self.value(x)?.shape().clone();
        let out = self.value(x)?.reshape(new_shape)?;
        self.record(TapeOp::Reshape { input_shape }, vec![x], out)
    }

    /// Differentiable axis transpose.
    pub fn transpose(&mut self, x: VarId, axis_a: usize, axis_b: usize) -> Result<VarId> {
        let out = self.value(x)?.transpose(axis_a, axis_b)?;
        self.record(TapeOp::Transpose { axis_a, axis_b }, vec![x], out)
    }

    /// Differentiable global sum.
    pub fn sum(&mut self, x: VarId) -> Result<VarId> {
        let input_shape = self.value(x)?.shape().clone();
        let out = self.value(x)?.sum()?;
        self.record(
            TapeOp::Sum {
                input_shape,
                axis: None,
                keepdim: false,
            },
            vec![x],
            out,
        )
    }

    /// Differentiable axis sum.
    pub fn sum_axis(&mut self, x: VarId, axis: usize, keepdim: bool) -> Result<VarId> {
        let input_shape = self.value(x)?.shape().clone();
        let out = self.value(x)?.sum_axis(axis, keepdim)?;
        self.record(
            TapeOp::Sum {
                input_shape,
                axis: Some(axis),
                keepdim,
            },
            vec![x],
            out,
        )
    }

    /// Differentiable global mean.
    pub fn mean(&mut self, x: VarId) -> Result<VarId> {
        let input_shape = self.value(x)?.shape().clone();
        let count = input_shape.numel();
        let out = self.value(x)?.mean()?;
        self.record(
            TapeOp::Mean {
                input_shape,
                axis: None,
                keepdim: false,
                count,
            },
            vec![x],
            out,
        )
    }

    /// Differentiable axis mean.
    pub fn mean_axis(&mut self, x: VarId, axis: usize, keepdim: bool) -> Result<VarId> {
        let input_shape = self.value(x)?.shape().clone();
        let count = input_shape.dim(axis)?;
        let out = self.value(x)?.mean_axis(axis, keepdim)?;
        self.record(
            TapeOp::Mean {
                input_shape,
                axis: Some(axis),
                keepdim,
                count,
            },
            vec![x],
            out,
        )
    }

    /// Differentiable 2D `im2col` unfolding `[N, C, H, W] -> [N, H_out * W_out, C * kH * kW]`.
    pub fn im2col(
        &mut self,
        x: VarId,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<VarId> {
        let x_val = self.value(x)?;
        let input_shape = x_val.shape().clone();
        let out = im2col_forward(x_val, kernel_h, kernel_w, stride, padding)?;
        self.record(
            TapeOp::Im2Col {
                input_shape,
                kernel_h,
                kernel_w,
                stride,
                padding,
            },
            vec![x],
            out,
        )
    }

    /// Executes a subgraph with **gradient checkpointing**.
    ///
    /// During the forward pass, intermediate activations are computed on a temporary
    /// scratch tape and immediately dropped; only the inputs and output are retained.
    /// During `backward`, the subgraph is recomputed on demand to propagate gradients.
    pub fn checkpoint<F>(&mut self, inputs: &[VarId], build: F) -> Result<VarId>
    where
        F: Fn(&mut Tape, &[VarId]) -> Result<VarId> + 'static,
    {
        let recompute: RecomputeFn = Rc::new(build);

        // Forward pass on an ephemeral no-grad scratch tape (intermediates are dropped)
        let mut scratch = Tape::new();
        scratch.set_grad_enabled(false);
        let mut local_inputs = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            let t = self.value(inp)?.clone();
            local_inputs.push(scratch.var(t, false));
        }
        let local_out = recompute(&mut scratch, &local_inputs)?;
        let out_tensor = scratch.value(local_out)?.clone();
        drop(scratch);

        self.record(TapeOp::Checkpoint { recompute }, inputs.to_vec(), out_tensor)
    }

    // --- Backpropagation ---

    /// Computes reverse-mode gradients of scalar `loss` with respect to all
    /// ancestor variables that require gradients.
    pub fn backward(&self, loss: VarId) -> Result<Gradients> {
        let mut grads = Gradients::new();
        self.backward_into(loss, &mut grads)?;
        Ok(grads)
    }

    /// Computes reverse-mode gradients of scalar `loss` and accumulates them
    /// into `grads` (supporting multi-step gradient accumulation).
    pub fn backward_into(&self, loss: VarId, grads: &mut Gradients) -> Result<()> {
        let loss_val = self.value(loss)?;
        if loss_val.numel() != 1 {
            return Err(Error::InvalidArgument {
                context: "Tape::backward",
                reason: format!(
                    "implicit backward requires a scalar loss (numel == 1), got shape {:?}",
                    loss_val.dims()
                ),
            });
        }
        let seed = Tensor::ones(loss_val.dims().to_vec(), loss_val.dtype())?;
        self.backward_with_seed_into(loss, seed, grads)
    }

    /// Computes reverse-mode vector-Jacobian products starting from an explicit
    /// upstream `seed` gradient tensor, accumulating into `grads`.
    pub fn backward_with_seed_into(
        &self,
        root: VarId,
        seed: Tensor,
        grads: &mut Gradients,
    ) -> Result<()> {
        let root_val = self.value(root)?;
        if root_val.shape() != seed.shape() {
            return Err(Error::ShapeMismatch {
                context: "Tape::backward_with_seed",
                expected: root_val.dims().to_vec(),
                actual: seed.dims().to_vec(),
            });
        }
        if root_val.dtype() != seed.dtype() {
            return Err(Error::DTypeMismatch {
                expected: root_val.dtype(),
                actual: seed.dtype(),
            });
        }

        // Local per-call cotangent map so repeated backward_into calls accumulate cleanly
        let mut local_grads = Gradients::new();
        local_grads.accumulate(root, seed)?;

        for idx in (0..=root.0).rev() {
            let var_id = VarId(idx);
            let Some(entry) = self.entries.get(idx) else {
                continue;
            };
            if !entry.requires_grad {
                continue;
            }
            let Some(grad_out) = local_grads.get(var_id).cloned() else {
                continue;
            };

            match &entry.op {
                TapeOp::Leaf => {}
                TapeOp::Add => {
                    let (a, b) = binary_inputs(&entry.inputs)?;
                    if self.requires_grad(a)? {
                        let ga = unbroadcast(&grad_out, self.value(a)?.shape())?;
                        local_grads.accumulate(a, ga)?;
                    }
                    if self.requires_grad(b)? {
                        let gb = unbroadcast(&grad_out, self.value(b)?.shape())?;
                        local_grads.accumulate(b, gb)?;
                    }
                }
                TapeOp::Sub => {
                    let (a, b) = binary_inputs(&entry.inputs)?;
                    if self.requires_grad(a)? {
                        let ga = unbroadcast(&grad_out, self.value(a)?.shape())?;
                        local_grads.accumulate(a, ga)?;
                    }
                    if self.requires_grad(b)? {
                        let gb = unbroadcast(&grad_out.neg()?, self.value(b)?.shape())?;
                        local_grads.accumulate(b, gb)?;
                    }
                }
                TapeOp::Mul => {
                    let (a, b) = binary_inputs(&entry.inputs)?;
                    if self.requires_grad(a)? {
                        let ga = unbroadcast(&grad_out.mul(self.value(b)?)?, self.value(a)?.shape())?;
                        local_grads.accumulate(a, ga)?;
                    }
                    if self.requires_grad(b)? {
                        let gb = unbroadcast(&grad_out.mul(self.value(a)?)?, self.value(b)?.shape())?;
                        local_grads.accumulate(b, gb)?;
                    }
                }
                TapeOp::Div => {
                    let (a, b) = binary_inputs(&entry.inputs)?;
                    let vb = self.value(b)?;
                    if self.requires_grad(a)? {
                        let ga = unbroadcast(&grad_out.div(vb)?, self.value(a)?.shape())?;
                        local_grads.accumulate(a, ga)?;
                    }
                    if self.requires_grad(b)? {
                        let va = self.value(a)?;
                        let b_sq = vb.mul(vb)?;
                        let raw_gb = grad_out.neg()?.mul(va)?.div(&b_sq)?;
                        let gb = unbroadcast(&raw_gb, vb.shape())?;
                        local_grads.accumulate(b, gb)?;
                    }
                }
                TapeOp::Neg => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        local_grads.accumulate(x, grad_out.neg()?)?;
                    }
                }
                TapeOp::Abs => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let sign = sign_mask(self.value(x)?)?;
                        local_grads.accumulate(x, grad_out.mul(&sign)?)?;
                    }
                }
                TapeOp::Exp => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        local_grads.accumulate(x, grad_out.mul(&entry.value)?)?;
                    }
                }
                TapeOp::Ln => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        local_grads.accumulate(x, grad_out.div(self.value(x)?)?)?;
                    }
                }
                TapeOp::Sqrt => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let two = Tensor::full(vec![], 2.0, entry.value.dtype())?;
                        let denom = entry.value.mul(&two)?;
                        local_grads.accumulate(x, grad_out.div(&denom)?)?;
                    }
                }
                TapeOp::Relu => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let mask = gt_zero_mask(self.value(x)?)?;
                        local_grads.accumulate(x, grad_out.mul(&mask)?)?;
                    }
                }
                TapeOp::Powf(p) => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let vx = self.value(x)?;
                        let p_scalar = Tensor::full(vec![], *p, vx.dtype())?;
                        let deriv = vx.powf(*p - 1.0)?.mul(&p_scalar)?;
                        local_grads.accumulate(x, grad_out.mul(&deriv)?)?;
                    }
                }
                TapeOp::MatMul => {
                    let (a, b) = binary_inputs(&entry.inputs)?;
                    let va = self.value(a)?;
                    let vb = self.value(b)?;
                    if self.requires_grad(a)? {
                        let b_rank = vb.rank();
                        let bt = vb.transpose(b_rank - 2, b_rank - 1)?;
                        let raw_ga = grad_out.matmul(&bt)?;
                        let ga = unbroadcast(&raw_ga, va.shape())?;
                        local_grads.accumulate(a, ga)?;
                    }
                    if self.requires_grad(b)? {
                        let a_rank = va.rank();
                        let at = va.transpose(a_rank - 2, a_rank - 1)?;
                        let raw_gb = at.matmul(&grad_out)?;
                        let gb = unbroadcast(&raw_gb, vb.shape())?;
                        local_grads.accumulate(b, gb)?;
                    }
                }
                TapeOp::Reshape { input_shape } => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        local_grads.accumulate(x, grad_out.reshape(input_shape.dims().to_vec())?)?;
                    }
                }
                TapeOp::Transpose { axis_a, axis_b } => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        local_grads.accumulate(x, grad_out.transpose(*axis_a, *axis_b)?)?;
                    }
                }
                TapeOp::Sum {
                    input_shape,
                    axis,
                    keepdim,
                } => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let expanded =
                            expand_reduced_grad(&grad_out, input_shape, *axis, *keepdim)?;
                        local_grads.accumulate(x, expanded)?;
                    }
                }
                TapeOp::Mean {
                    input_shape,
                    axis,
                    keepdim,
                    count,
                } => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let scale = Tensor::full(vec![], 1.0 / (*count as f64), grad_out.dtype())?;
                        let scaled = grad_out.mul(&scale)?;
                        let expanded =
                            expand_reduced_grad(&scaled, input_shape, *axis, *keepdim)?;
                        local_grads.accumulate(x, expanded)?;
                    }
                }
                TapeOp::Im2Col {
                    input_shape,
                    kernel_h,
                    kernel_w,
                    stride,
                    padding,
                } => {
                    let x = unary_input(&entry.inputs)?;
                    if self.requires_grad(x)? {
                        let dx = col2im_backward(
                            &grad_out,
                            input_shape,
                            *kernel_h,
                            *kernel_w,
                            *stride,
                            *padding,
                        )?;
                        local_grads.accumulate(x, dx)?;
                    }
                }
                TapeOp::Checkpoint { recompute } => {
                    // Recompute local subgraph with grad tracking enabled
                    let mut local_tape = Tape::new();
                    let mut local_inputs = Vec::with_capacity(entry.inputs.len());
                    for &inp in &entry.inputs {
                        let t = self.value(inp)?.clone();
                        let req = self.requires_grad(inp)?;
                        local_inputs.push(local_tape.var(t, req));
                    }
                    let local_out = recompute(&mut local_tape, &local_inputs)?;
                    let mut sub_grads = Gradients::new();
                    local_tape.backward_with_seed_into(local_out, grad_out, &mut sub_grads)?;

                    for (&orig_inp, &loc_inp) in entry.inputs.iter().zip(local_inputs.iter()) {
                        if let Some(g) = sub_grads.get(loc_inp).cloned() {
                            local_grads.accumulate(orig_inp, g)?;
                        }
                    }
                }
            }
        }

        // Accumulate computed gradients into caller's Gradients map
        for (var_id, g) in local_grads.grads {
            grads.accumulate(var_id, g)?;
        }
        Ok(())
    }
}

fn unary_input(inputs: &[VarId]) -> Result<VarId> {
    inputs.first().copied().ok_or(Error::InternalInvariant {
        message: "unary tape entry missing input",
    })
}

fn binary_inputs(inputs: &[VarId]) -> Result<(VarId, VarId)> {
    let a = inputs.first().copied().ok_or(Error::InternalInvariant {
        message: "binary tape entry missing lhs",
    })?;
    let b = inputs.get(1).copied().ok_or(Error::InternalInvariant {
        message: "binary tape entry missing rhs",
    })?;
    Ok((a, b))
}

/// Reduces a gradient tensor back to `target_shape` by summing over broadcasted axes.
pub(crate) fn unbroadcast(grad: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    if grad.shape() == target_shape {
        return Ok(grad.clone());
    }
    let mut cur = grad.clone();
    // 1. Sum out leading prepended dimensions
    while cur.rank() > target_shape.rank() {
        cur = cur.sum_axis(0, false)?;
    }
    // 2. Sum out dimensions that were size-1 in target_shape
    for (axis, &t_dim) in target_shape.dims().iter().enumerate() {
        if t_dim == 1 && cur.shape().dim(axis)? > 1 {
            cur = cur.sum_axis(axis, true)?;
        }
    }
    Ok(cur)
}

fn expand_reduced_grad(
    grad: &Tensor,
    input_shape: &Shape,
    axis: Option<usize>,
    keepdim: bool,
) -> Result<Tensor> {
    let reshaped = match axis {
        None => grad.reshape(vec![1; input_shape.rank()])?,
        Some(ax) => {
            if keepdim {
                grad.clone()
            } else {
                grad.unsqueeze(ax)?
            }
        }
    };
    reshaped
        .broadcast_to(input_shape.dims().to_vec())?
        .contiguous()
}

fn gt_zero_mask(x: &Tensor) -> Result<Tensor> {
    match x.dtype() {
        DType::F32 => {
            let vals: Vec<f32> = x
                .to_f32_vec()?
                .into_iter()
                .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_f32(&vals, x.dims().to_vec())
        }
        DType::F64 => {
            let vals: Vec<f64> = x
                .to_f64_vec()?
                .into_iter()
                .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_f64(&vals, x.dims().to_vec())
        }
        dtype => Err(Error::UnsupportedDType {
            op: "relu_backward",
            dtype,
        }),
    }
}

fn sign_mask(x: &Tensor) -> Result<Tensor> {
    match x.dtype() {
        DType::F32 => {
            let vals: Vec<f32> = x
                .to_f32_vec()?
                .into_iter()
                .map(|v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            Tensor::from_f32(&vals, x.dims().to_vec())
        }
        DType::F64 => {
            let vals: Vec<f64> = x
                .to_f64_vec()?
                .into_iter()
                .map(|v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            Tensor::from_f64(&vals, x.dims().to_vec())
        }
        dtype => Err(Error::UnsupportedDType {
            op: "abs_backward",
            dtype,
        }),
    }
}

fn im2col_forward(
    x: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(Error::InvalidRank {
            context: "im2col_forward",
            rank: x.rank(),
            expected: 4,
        });
    }
    let n = x.shape().dim(0)?;
    let c_in = x.shape().dim(1)?;
    let h_in = x.shape().dim(2)?;
    let w_in = x.shape().dim(3)?;
    let h_pad = h_in + 2 * padding;
    let w_pad = w_in + 2 * padding;
    if stride == 0 || kernel_h == 0 || kernel_w == 0 || h_pad < kernel_h || w_pad < kernel_w {
        return Err(Error::InvalidArgument {
            context: "im2col_forward",
            reason: "invalid kernel/stride/padding for input spatial dimensions".to_owned(),
        });
    }
    let h_out = (h_pad - kernel_h) / stride + 1;
    let w_out = (w_pad - kernel_w) / stride + 1;
    let k_col = c_in * kernel_h * kernel_w;
    let spatial_out = h_out * w_out;
    let dtype = x.dtype();
    let x_f64 = x.to_dtype(DType::F64)?.to_f64_vec()?;
    let mut col = vec![0.0f64; n * spatial_out * k_col];

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = batch * spatial_out + oh * w_out + ow;
                let mut col_idx = 0usize;
                for ic in 0..c_in {
                    for kh in 0..kernel_h {
                        let ih_s = (oh * stride + kh) as isize - (padding as isize);
                        for kw in 0..kernel_w {
                            let iw_s = (ow * stride + kw) as isize - (padding as isize);
                            let val = if ih_s >= 0
                                && (ih_s as usize) < h_in
                                && iw_s >= 0
                                && (iw_s as usize) < w_in
                            {
                                let src = ((batch * c_in + ic) * h_in + (ih_s as usize)) * w_in
                                    + (iw_s as usize);
                                x_f64.get(src).copied().unwrap_or(0.0)
                            } else {
                                0.0
                            };
                            if let Some(slot) = col.get_mut(row * k_col + col_idx) {
                                *slot = val;
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    let out_f64 = Tensor::from_f64(&col, vec![n, spatial_out, k_col])?;
    if dtype == DType::F64 {
        Ok(out_f64)
    } else {
        out_f64.to_dtype(dtype)
    }
}

fn col2im_backward(
    grad_col: &Tensor,
    input_shape: &Shape,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let n = input_shape.dim(0)?;
    let c_in = input_shape.dim(1)?;
    let h_in = input_shape.dim(2)?;
    let w_in = input_shape.dim(3)?;
    let h_out = (h_in + 2 * padding - kernel_h) / stride + 1;
    let w_out = (w_in + 2 * padding - kernel_w) / stride + 1;
    let k_col = c_in * kernel_h * kernel_w;
    let spatial_out = h_out * w_out;

    let dtype = grad_col.dtype();
    let g_f64 = grad_col.to_dtype(DType::F64)?.to_f64_vec()?;
    let mut dx = vec![0.0f64; n * c_in * h_in * w_in];

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = batch * spatial_out + oh * w_out + ow;
                let mut col_idx = 0usize;
                for ic in 0..c_in {
                    for kh in 0..kernel_h {
                        let ih_s = (oh * stride + kh) as isize - (padding as isize);
                        for kw in 0..kernel_w {
                            let iw_s = (ow * stride + kw) as isize - (padding as isize);
                            if ih_s >= 0
                                && (ih_s as usize) < h_in
                                && iw_s >= 0
                                && (iw_s as usize) < w_in
                            {
                                let dst = ((batch * c_in + ic) * h_in + (ih_s as usize)) * w_in
                                    + (iw_s as usize);
                                let gv = g_f64.get(row * k_col + col_idx).copied().unwrap_or(0.0);
                                if let Some(slot) = dx.get_mut(dst) {
                                    *slot += gv;
                                }
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    let dx_f64 = Tensor::from_f64(&dx, vec![n, c_in, h_in, w_in])?;
    if dtype == DType::F64 {
        Ok(dx_f64)
    } else {
        dx_f64.to_dtype(dtype)
    }
}
