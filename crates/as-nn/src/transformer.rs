//! Multi-Head Self-Attention and Pre-Norm Transformer Block.

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, Result};
use as_tensor::Tensor;

use crate::activations::{gelu, softmax};
use crate::init::Rng;
use crate::layers::{LayerNorm, Linear};
use crate::module::{Module, Param};

/// Multi-Head Self-Attention layer supporting optional causal masking.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Query projection `[embed_dim, embed_dim]`.
    pub w_q: Linear,
    /// Key projection `[embed_dim, embed_dim]`.
    pub w_k: Linear,
    /// Value projection `[embed_dim, embed_dim]`.
    pub w_v: Linear,
    /// Output projection `[embed_dim, embed_dim]`.
    pub w_o: Linear,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Creates a new [`MultiHeadAttention`] layer.
    pub fn new(
        prefix: &str,
        embed_dim: usize,
        num_heads: usize,
        dtype: DType,
        rng: &mut Rng,
    ) -> Result<Self> {
        if num_heads == 0 || embed_dim % num_heads != 0 {
            return Err(Error::InvalidArgument {
                context: "MultiHeadAttention::new",
                reason: format!(
                    "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                ),
            });
        }
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            w_q: Linear::new(&format!("{prefix}.w_q"), embed_dim, embed_dim, false, dtype, rng)?,
            w_k: Linear::new(&format!("{prefix}.w_k"), embed_dim, embed_dim, false, dtype, rng)?,
            w_v: Linear::new(&format!("{prefix}.w_v"), embed_dim, embed_dim, false, dtype, rng)?,
            w_o: Linear::new(&format!("{prefix}.w_o"), embed_dim, embed_dim, false, dtype, rng)?,
            embed_dim,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass for `x` of shape `[batch, seq_len, embed_dim]`.
    pub fn forward(
        &self,
        tape: &mut Tape,
        x: VarId,
        causal: bool,
    ) -> Result<(VarId, Vec<VarId>)> {
        let x_val = tape.value(x)?;
        if x_val.rank() != 3 {
            return Err(Error::InvalidRank {
                context: "MultiHeadAttention::forward",
                rank: x_val.rank(),
                expected: 3,
            });
        }
        let batch = x_val.shape().dim(0)?;
        let seq = x_val.shape().dim(1)?;
        let dim = x_val.shape().dim(2)?;
        if dim != self.embed_dim {
            return Err(Error::ShapeMismatch {
                context: "MultiHeadAttention::forward",
                expected: vec![batch, seq, self.embed_dim],
                actual: x_val.dims().to_vec(),
            });
        }
        let dtype = x_val.dtype();

        let (q_lin, p_q) = self.w_q.forward(tape, x)?;
        let (k_lin, p_k) = self.w_k.forward(tape, x)?;
        let (v_lin, p_v) = self.w_v.forward(tape, x)?;

        // [B, T, D] -> [B, T, H, Hd] -> [B, H, T, Hd]
        let q_4d = tape.reshape(q_lin, vec![batch, seq, self.num_heads, self.head_dim])?;
        let q = tape.transpose(q_4d, 1, 2)?;
        let k_4d = tape.reshape(k_lin, vec![batch, seq, self.num_heads, self.head_dim])?;
        let k = tape.transpose(k_4d, 1, 2)?;
        let v_4d = tape.reshape(v_lin, vec![batch, seq, self.num_heads, self.head_dim])?;
        let v = tape.transpose(v_4d, 1, 2)?;

        let kt = tape.transpose(k, 2, 3)?; // [B, H, Hd, T]
        let scale = tape.var(
            Tensor::full(vec![], 1.0 / (self.head_dim as f64).sqrt(), dtype)?,
            false,
        );
        let qk = tape.matmul(q, kt)?;
        let mut scores = tape.mul(qk, scale)?; // [B, H, T, T]

        if causal && seq > 1 {
            let mut mask = vec![0.0f64; seq * seq];
            for i in 0..seq {
                for j in (i + 1)..seq {
                    if let Some(slot) = mask.get_mut(i * seq + j) {
                        *slot = -1e9;
                    }
                }
            }
            let m_f64 = Tensor::from_f64(&mask, vec![1, 1, seq, seq])?;
            let m_t = if dtype == DType::F64 {
                m_f64
            } else {
                m_f64.to_dtype(dtype)?
            };
            let m_var = tape.var(m_t, false);
            scores = tape.add(scores, m_var)?;
        }

        let attn = softmax(tape, scores, 3)?;
        let ctx = tape.matmul(attn, v)?; // [B, H, T, Hd]
        let ctx_t = tape.transpose(ctx, 1, 2)?; // [B, T, H, Hd]
        let merged = tape.reshape(ctx_t, vec![batch, seq, self.embed_dim])?;

        let (out, p_o) = self.w_o.forward(tape, merged)?;
        let mut params = Vec::with_capacity(4);
        params.extend(p_q);
        params.extend(p_k);
        params.extend(p_v);
        params.extend(p_o);
        Ok((out, params))
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = Vec::new();
        p.extend(self.w_q.parameters());
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p.extend(self.w_o.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = Vec::new();
        p.extend(self.w_q.parameters_mut());
        p.extend(self.w_k.parameters_mut());
        p.extend(self.w_v.parameters_mut());
        p.extend(self.w_o.parameters_mut());
        p
    }
}

/// Standard Pre-Norm Transformer block:
/// `x = x + MHA(LN1(x))`, `x = x + FFN(LN2(x))`.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Pre-attention layer normalization.
    pub ln1: LayerNorm,
    /// Multi-head self-attention.
    pub attn: MultiHeadAttention,
    /// Pre-FFN layer normalization.
    pub ln2: LayerNorm,
    /// Feed-forward expansion layer.
    pub ff_up: Linear,
    /// Feed-forward projection layer.
    pub ff_down: Linear,
}

impl TransformerBlock {
    /// Creates a new [`TransformerBlock`].
    pub fn new(
        prefix: &str,
        embed_dim: usize,
        num_heads: usize,
        ff_hidden_dim: usize,
        dtype: DType,
        rng: &mut Rng,
    ) -> Result<Self> {
        Ok(Self {
            ln1: LayerNorm::new(&format!("{prefix}.ln1"), embed_dim, 1e-5, dtype)?,
            attn: MultiHeadAttention::new(
                &format!("{prefix}.attn"),
                embed_dim,
                num_heads,
                dtype,
                rng,
            )?,
            ln2: LayerNorm::new(&format!("{prefix}.ln2"), embed_dim, 1e-5, dtype)?,
            ff_up: Linear::new(
                &format!("{prefix}.ff_up"),
                embed_dim,
                ff_hidden_dim,
                true,
                dtype,
                rng,
            )?,
            ff_down: Linear::new(
                &format!("{prefix}.ff_down"),
                ff_hidden_dim,
                embed_dim,
                true,
                dtype,
                rng,
            )?,
        })
    }

    /// Forward pass for `x` of shape `[batch, seq_len, embed_dim]`.
    pub fn forward(&self, tape: &mut Tape, x: VarId, causal: bool) -> Result<(VarId, Vec<VarId>)> {
        let (h1, p1) = self.ln1.forward(tape, x)?;
        let (a_out, p_attn) = self.attn.forward(tape, h1, causal)?;
        let res1 = tape.add(x, a_out)?;

        let (h2, p2) = self.ln2.forward(tape, res1)?;
        let (up, p_up) = self.ff_up.forward(tape, h2)?;
        let act = gelu(tape, up)?;
        let (down, p_down) = self.ff_down.forward(tape, act)?;
        let out = tape.add(res1, down)?;

        let mut params = Vec::new();
        params.extend(p1);
        params.extend(p_attn);
        params.extend(p2);
        params.extend(p_up);
        params.extend(p_down);
        Ok((out, params))
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = Vec::new();
        p.extend(self.ln1.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ff_up.parameters());
        p.extend(self.ff_down.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = Vec::new();
        p.extend(self.ln1.parameters_mut());
        p.extend(self.attn.parameters_mut());
        p.extend(self.ln2.parameters_mut());
        p.extend(self.ff_up.parameters_mut());
        p.extend(self.ff_down.parameters_mut());
        p
    }
}
