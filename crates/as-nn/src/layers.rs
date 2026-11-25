//! Core neural network layers: [`Linear`], [`Embedding`], [`LayerNorm`],
//! [`RmsNorm`], [`Dropout`], and [`Conv2d`].

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, Result};
use as_tensor::Tensor;

use crate::init::{Rng, kaiming_normal, xavier_uniform};
use crate::module::{Module, Param};

/// Fully-connected affine transformation: `y = x @ W^T + b`.
///
/// Supports arbitrary leading batch/sequence dimensions `[..., in_features] -> [..., out_features]`.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix of shape `[out_features, in_features]`.
    pub weight: Param,
    /// Optional bias vector of shape `[out_features]`.
    pub bias: Option<Param>,
}

impl Linear {
    /// Creates a new [`Linear`] layer with Xavier uniform weight initialization.
    pub fn new(
        prefix: &str,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        dtype: DType,
        rng: &mut Rng,
    ) -> Result<Self> {
        let w = xavier_uniform(vec![out_features, in_features], in_features, out_features, dtype, rng)?;
        let bias = if use_bias {
            Some(Param::new(
                format!("{prefix}.bias"),
                Tensor::zeros(vec![out_features], dtype)?,
            ))
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(format!("{prefix}.weight"), w),
            bias,
        })
    }

    /// Forward pass returning the output [`VarId`] and the registered parameter [`VarId`]s
    /// `(out, weight_var, bias_var)`.
    pub fn forward(&self, tape: &mut Tape, x: VarId) -> Result<(VarId, Vec<VarId>)> {
        let w_var = self.weight.register(tape);
        let wt = tape.transpose(w_var, 0, 1)?;
        let mm = tape.matmul(x, wt)?;
        let mut reg = vec![w_var];
        let out = if let Some(b) = &self.bias {
            let b_var = b.register(tape);
            reg.push(b_var);
            tape.add(mm, b_var)?
        } else {
            mm
        };
        Ok((out, reg))
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = vec![&self.weight];
        if let Some(b) = &self.bias {
            p.push(b);
        }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = vec![&mut self.weight];
        if let Some(b) = &mut self.bias {
            p.push(b);
        }
        p
    }
}

/// Token / discrete-index embedding lookup table of shape `[vocab_size, embed_dim]`.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding matrix of shape `[vocab_size, embed_dim]`.
    pub weight: Param,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    /// Creates a new [`Embedding`] table initialized from `N(0, 0.02)`.
    pub fn new(
        prefix: &str,
        vocab_size: usize,
        embed_dim: usize,
        dtype: DType,
        rng: &mut Rng,
    ) -> Result<Self> {
        let mut vals = Vec::with_capacity(vocab_size * embed_dim);
        for _ in 0..(vocab_size * embed_dim) {
            vals.push(rng.normal(0.0, 0.02));
        }
        let t = Tensor::from_f64(&vals, vec![vocab_size, embed_dim])?;
        let w = if dtype == DType::F64 {
            t
        } else {
            t.to_dtype(dtype)?
        };
        Ok(Self {
            weight: Param::new(format!("{prefix}.weight"), w),
            vocab_size,
            embed_dim,
        })
    }

    /// Differentiable lookup for token IDs of shape `[batch, seq_len]` -> `[batch, seq_len, embed_dim]`.
    pub fn forward(
        &self,
        tape: &mut Tape,
        indices: &[usize],
        batch: usize,
        seq_len: usize,
    ) -> Result<(VarId, Vec<VarId>)> {
        let total = batch * seq_len;
        if indices.len() != total {
            return Err(Error::BufferLengthMismatch {
                expected: total,
                actual: indices.len(),
            });
        }
        let dtype = self.weight.dtype();
        let mut one_hot = vec![0.0f64; total * self.vocab_size];
        for (pos, &tok) in indices.iter().enumerate() {
            if tok >= self.vocab_size {
                return Err(Error::IndexOutOfBounds {
                    index: tok,
                    len: self.vocab_size,
                });
            }
            if let Some(slot) = one_hot.get_mut(pos * self.vocab_size + tok) {
                *slot = 1.0;
            }
        }
        let oh_f64 = Tensor::from_f64(&one_hot, vec![total, self.vocab_size])?;
        let oh_t = if dtype == DType::F64 {
            oh_f64
        } else {
            oh_f64.to_dtype(dtype)?
        };
        let oh_var = tape.var(oh_t, false);
        let w_var = self.weight.register(tape);
        let flat_out = tape.matmul(oh_var, w_var)?;
        let out = tape.reshape(flat_out, vec![batch, seq_len, self.embed_dim])?;
        Ok((out, vec![w_var]))
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<&Param> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.weight]
    }
}

/// Layer Normalization over the last dimension.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter `gamma` of shape `[dim]`.
    pub gamma: Param,
    /// Shift parameter `beta` of shape `[dim]`.
    pub beta: Param,
    /// Numerical stability epsilon.
    pub eps: f64,
}

impl LayerNorm {
    /// Creates a new [`LayerNorm`] layer.
    pub fn new(prefix: &str, dim: usize, eps: f64, dtype: DType) -> Result<Self> {
        Ok(Self {
            gamma: Param::new(format!("{prefix}.gamma"), Tensor::ones(vec![dim], dtype)?),
            beta: Param::new(format!("{prefix}.beta"), Tensor::zeros(vec![dim], dtype)?),
            eps,
        })
    }

    /// Forward pass normalizing along the last axis.
    pub fn forward(&self, tape: &mut Tape, x: VarId) -> Result<(VarId, Vec<VarId>)> {
        let x_val = tape.value(x)?;
        let rank = x_val.rank();
        if rank == 0 {
            return Err(Error::InvalidRank {
                context: "LayerNorm::forward",
                rank: 0,
                expected: 1,
            });
        }
        let last_axis = rank - 1;
        let dtype = x_val.dtype();

        let mean = tape.mean_axis(x, last_axis, true)?;
        let centered = tape.sub(x, mean)?;
        let sq = tape.mul(centered, centered)?;
        let var = tape.mean_axis(sq, last_axis, true)?;
        let eps_var = tape.var(Tensor::full(vec![], self.eps, dtype)?, false);
        let var_eps = tape.add(var, eps_var)?;
        let std = tape.sqrt(var_eps)?;
        let norm = tape.div(centered, std)?;

        let g_var = self.gamma.register(tape);
        let b_var = self.beta.register(tape);
        let scaled = tape.mul(norm, g_var)?;
        let out = tape.add(scaled, b_var)?;
        Ok((out, vec![g_var, b_var]))
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<&Param> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

/// Root Mean Square Normalization (`x / sqrt(mean(x^2) + eps) * weight`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Scale parameter of shape `[dim]`.
    pub weight: Param,
    /// Numerical stability epsilon.
    pub eps: f64,
}

impl RmsNorm {
    /// Creates a new [`RmsNorm`] layer.
    pub fn new(prefix: &str, dim: usize, eps: f64, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: Param::new(format!("{prefix}.weight"), Tensor::ones(vec![dim], dtype)?),
            eps,
        })
    }

    /// Forward pass normalizing along the last axis.
    pub fn forward(&self, tape: &mut Tape, x: VarId) -> Result<(VarId, Vec<VarId>)> {
        let x_val = tape.value(x)?;
        let rank = x_val.rank();
        if rank == 0 {
            return Err(Error::InvalidRank {
                context: "RmsNorm::forward",
                rank: 0,
                expected: 1,
            });
        }
        let last_axis = rank - 1;
        let dtype = x_val.dtype();

        let sq = tape.mul(x, x)?;
        let ms = tape.mean_axis(sq, last_axis, true)?;
        let eps_var = tape.var(Tensor::full(vec![], self.eps, dtype)?, false);
        let ms_eps = tape.add(ms, eps_var)?;
        let rms = tape.sqrt(ms_eps)?;
        let norm = tape.div(x, rms)?;

        let w_var = self.weight.register(tape);
        let out = tape.mul(norm, w_var)?;
        Ok((out, vec![w_var]))
    }
}

impl Module for RmsNorm {
    fn parameters(&self) -> Vec<&Param> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.weight]
    }
}

/// Deterministic inverted Dropout layer.
#[derive(Debug, Clone, Copy)]
pub struct Dropout {
    /// Drop probability in `[0.0, 1.0)`.
    pub p: f64,
    /// Whether the layer is in training mode.
    pub training: bool,
}

impl Dropout {
    /// Creates a new [`Dropout`] layer.
    pub fn new(p: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                context: "Dropout::new",
                reason: format!("dropout probability p must be in [0.0, 1.0), got {p}"),
            });
        }
        Ok(Self { p, training: true })
    }

    /// Applies inverted dropout during training, or identity during evaluation.
    pub fn forward(&self, tape: &mut Tape, x: VarId, rng: &mut Rng) -> Result<VarId> {
        if !self.training || self.p == 0.0 {
            return Ok(x);
        }
        let val = tape.value(x)?;
        let numel = val.numel();
        let dtype = val.dtype();
        let scale = 1.0 / (1.0 - self.p);
        let mut mask_vals = Vec::with_capacity(numel);
        for _ in 0..numel {
            mask_vals.push(if rng.next_f64() < self.p { 0.0 } else { scale });
        }
        let mask_f64 = Tensor::from_f64(&mask_vals, val.dims().to_vec())?;
        let mask_t = if dtype == DType::F64 {
            mask_f64
        } else {
            mask_f64.to_dtype(dtype)?
        };
        let mask = tape.var(mask_t, false);
        tape.mul(x, mask)
    }
}

/// 2D Convolution layer (`NCHW` format) implemented via `im2col` + tiled GEMM.
#[derive(Debug, Clone)]
pub struct Conv2d {
    /// Weight tensor of shape `[out_channels, in_channels, kernel_h, kernel_w]`.
    pub weight: Param,
    /// Optional bias of shape `[1, out_channels, 1, 1]`.
    pub bias: Option<Param>,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    /// Creates a new [`Conv2d`] layer with Kaiming normal initialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        dtype: DType,
        rng: &mut Rng,
    ) -> Result<Self> {
        if stride == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(Error::InvalidArgument {
                context: "Conv2d::new",
                reason: "kernel dimensions and stride must be >= 1".to_owned(),
            });
        }
        let fan_in = in_channels * kernel_h * kernel_w;
        let w = kaiming_normal(
            vec![out_channels, in_channels, kernel_h, kernel_w],
            fan_in,
            dtype,
            rng,
        )?;
        let bias = if use_bias {
            Some(Param::new(
                format!("{prefix}.bias"),
                Tensor::zeros(vec![1, out_channels, 1, 1], dtype)?,
            ))
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(format!("{prefix}.weight"), w),
            bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        })
    }

    /// Forward pass for `x` of shape `[N, C_in, H, W]` -> `[N, C_out, H_out, W_out]`.
    pub fn forward(&self, tape: &mut Tape, x: VarId) -> Result<(VarId, Vec<VarId>)> {
        let x_val = tape.value(x)?;
        if x_val.rank() != 4 {
            return Err(Error::InvalidRank {
                context: "Conv2d::forward",
                rank: x_val.rank(),
                expected: 4,
            });
        }
        let n = x_val.shape().dim(0)?;
        let c_in = x_val.shape().dim(1)?;
        let h_in = x_val.shape().dim(2)?;
        let w_in = x_val.shape().dim(3)?;
        if c_in != self.in_channels {
            return Err(Error::ShapeMismatch {
                context: "Conv2d::forward in_channels",
                expected: vec![n, self.in_channels, h_in, w_in],
                actual: x_val.dims().to_vec(),
            });
        }
        let h_pad = h_in + 2 * self.padding;
        let w_pad = w_in + 2 * self.padding;
        if h_pad < self.kernel_h || w_pad < self.kernel_w {
            return Err(Error::InvalidArgument {
                context: "Conv2d::forward",
                reason: "input spatial dimensions smaller than kernel".to_owned(),
            });
        }
        let h_out = (h_pad - self.kernel_h) / self.stride + 1;
        let w_out = (w_pad - self.kernel_w) / self.stride + 1;
        let k_col = self.in_channels * self.kernel_h * self.kernel_w;

        // Differentiable im2col unfolding [n, spatial_out, k_col]
        let col_var = tape.im2col(x, self.kernel_h, self.kernel_w, self.stride, self.padding)?;

        let w_var = self.weight.register(tape);
        let w_2d = tape.reshape(w_var, vec![self.out_channels, k_col])?;
        let w_t = tape.transpose(w_2d, 0, 1)?; // [k_col, out_channels]
        // [n, spatial_out, k_col] @ [k_col, out_channels] -> [n, spatial_out, out_channels]
        let mm = tape.matmul(col_var, w_t)?;
        let nchw_flat = tape.transpose(mm, 1, 2)?; // [n, out_channels, spatial_out]
        let mut out = tape.reshape(nchw_flat, vec![n, self.out_channels, h_out, w_out])?;

        let mut reg = vec![w_var];
        if let Some(b) = &self.bias {
            let b_var = b.register(tape);
            reg.push(b_var);
            out = tape.add(out, b_var)?;
        }
        Ok((out, reg))
    }
}

impl Module for Conv2d {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = vec![&self.weight];
        if let Some(b) = &self.bias {
            p.push(b);
        }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = vec![&mut self.weight];
        if let Some(b) = &mut self.bias {
            p.push(b);
        }
        p
    }
}
