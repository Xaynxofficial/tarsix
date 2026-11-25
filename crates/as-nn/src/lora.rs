//! Low-Rank Adaptation ([`LoRALinear`]) for parameter-efficient fine-tuning (Hu et al., 2021).
//!
//! Freezes the pretrained base weight `W [out, in]` and trains only two low-rank matrices
//! `A [rank, in]` and `B [out, rank]` (zero-initialized so initial delta is identically 0):
//!
//! `y = x @ W^T + (alpha / rank) * ((x @ A^T) @ B^T)`
//!
//! After fine-tuning, [`LoRALinear::merge_weights`] folds `(alpha / rank) * (B @ A)` back
//! into `W` for zero-overhead inference.

use as_autograd::{Tape, VarId};
use as_core::{Error, Result};
use as_tensor::Tensor;

use crate::init::{Rng, kaiming_normal};
use crate::layers::Linear;
use crate::module::{Module, Param};

/// Parameter-efficient Low-Rank Adaptation wrapper around a frozen [`Linear`] weight.
#[derive(Debug, Clone)]
pub struct LoRALinear {
    /// Frozen pretrained weight `[out_features, in_features]`.
    base_weight: Tensor,
    /// Optional frozen pretrained bias `[out_features]`.
    base_bias: Option<Tensor>,
    /// Trainable down-projection `A [rank, in_features]` (Kaiming initialized).
    pub lora_a: Param,
    /// Trainable up-projection `B [out_features, rank]` (zero initialized).
    pub lora_b: Param,
    /// Low-rank bottleneck dimension `r`.
    pub rank: usize,
    /// LoRA scaling numerator `alpha`.
    pub alpha: f64,
    /// Effective scale factor `alpha / rank`.
    pub scaling: f64,
    prefix: String,
}

impl LoRALinear {
    /// Wraps an existing pretrained [`Linear`] layer with trainable LoRA adapters `(A, B)`.
    pub fn from_linear(
        prefix: &str,
        base: &Linear,
        rank: usize,
        alpha: f64,
        rng: &mut Rng,
    ) -> Result<Self> {
        if rank == 0 {
            return Err(Error::InvalidArgument {
                context: "LoRALinear::from_linear",
                reason: "LoRA rank must be >= 1".to_owned(),
            });
        }
        let out_features = base.weight.shape().dim(0)?;
        let in_features = base.weight.shape().dim(1)?;
        let dtype = base.weight.dtype();

        let a_data = kaiming_normal(vec![rank, in_features], in_features, dtype, rng)?;
        let b_data = Tensor::zeros(vec![out_features, rank], dtype)?;

        Ok(Self {
            base_weight: base.weight.data().clone(),
            base_bias: base.bias.as_ref().map(|b| b.data().clone()),
            lora_a: Param::new(format!("{prefix}.lora_a"), a_data),
            lora_b: Param::new(format!("{prefix}.lora_b"), b_data),
            rank,
            alpha,
            scaling: alpha / (rank as f64),
            prefix: prefix.to_owned(),
        })
    }

    /// Fraction of trainable parameters relative to the frozen base weight (`(A + B) / W`).
    pub fn trainable_param_ratio(&self) -> f64 {
        let base_n = self.base_weight.numel().max(1) as f64;
        let lora_n = (self.lora_a.data().numel() + self.lora_b.data().numel()) as f64;
        lora_n / base_n
    }

    /// Borrows the frozen base weight tensor.
    pub fn base_weight(&self) -> &Tensor {
        &self.base_weight
    }

    /// Forward pass: `base(x) + scaling * ((x @ A^T) @ B^T)`.
    ///
    /// Returns `(output_var, [lora_a_var, lora_b_var])`. The frozen base weight is
    /// registered with `requires_grad = false`, so no gradients or optimizer states
    /// are allocated for it.
    pub fn forward(&self, tape: &mut Tape, x: VarId) -> Result<(VarId, Vec<VarId>)> {
        let dtype = self.base_weight.dtype();

        // 1. Frozen base projection: x @ W^T (+ b)
        let w_var = tape.var(self.base_weight.clone(), false);
        let wt = tape.transpose(w_var, 0, 1)?;
        let mut base_out = tape.matmul(x, wt)?;
        if let Some(b) = &self.base_bias {
            let b_var = tape.var(b.clone(), false);
            base_out = tape.add(base_out, b_var)?;
        }

        // 2. Trainable low-rank branch: ((x @ A^T) @ B^T) * scaling
        let a_var = self.lora_a.register(tape);
        let b_var = self.lora_b.register(tape);
        let at = tape.transpose(a_var, 0, 1)?;
        let bt = tape.transpose(b_var, 0, 1)?;

        let down = tape.matmul(x, at)?;
        let up = tape.matmul(down, bt)?;
        let scale_var = tape.var(Tensor::full(vec![], self.scaling, dtype)?, false);
        let delta = tape.mul(up, scale_var)?;

        let out = tape.add(base_out, delta)?;
        Ok((out, vec![a_var, b_var]))
    }

    /// Folds the trained LoRA adapters permanently into the base weight:
    /// `W_merged = W_base + scaling * (B @ A)`, returning a standard zero-overhead [`Linear`].
    pub fn merge_weights(&self) -> Result<Linear> {
        let dtype = self.base_weight.dtype();
        let ba = self.lora_b.data().matmul(self.lora_a.data())?;
        let scale_t = Tensor::full(vec![], self.scaling, dtype)?;
        let merged_w = self.base_weight.add(&ba.mul(&scale_t)?)?;
        let merged_bias = self
            .base_bias
            .as_ref()
            .map(|b| Param::new(format!("{}.bias", self.prefix), b.clone()));
        Ok(Linear {
            weight: Param::new(format!("{}.weight", self.prefix), merged_w),
            bias: merged_bias,
        })
    }
}

impl Module for LoRALinear {
    fn parameters(&self) -> Vec<&Param> {
        vec![&self.lora_a, &self.lora_b]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }
}
