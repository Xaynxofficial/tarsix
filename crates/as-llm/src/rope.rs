//! Rotary Positional Embeddings (RoPE).

use as_core::{Error, Result};
use as_tensor::Tensor;

/// Precomputed `cos` and `sin` tables for Rotary Positional Embeddings.
#[derive(Debug, Clone)]
pub struct RoPECache {
    max_seq_len: usize,
    head_dim: usize,
    half_dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RoPECache {
    /// Precomputes RoPE frequencies for `max_seq_len` positions and even `head_dim`.
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f64) -> Result<Self> {
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::InvalidArgument {
                context: "RoPECache::new",
                reason: format!("head_dim must be a positive even integer, got {head_dim}"),
            });
        }
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exp = (2.0 * (i as f64)) / (head_dim as f64);
            inv_freq.push(1.0 / theta.powf(exp));
        }

        let mut cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin = Vec::with_capacity(max_seq_len * half_dim);
        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = (pos as f64) * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }

        Ok(Self {
            max_seq_len,
            head_dim,
            half_dim,
            cos,
            sin,
        })
    }

    /// Applies RoPE to an `f32` tensor `x` of shape `[batch, num_heads, seq_len, head_dim]`
    /// starting at sequence position `pos_offset`.
    pub fn apply(&self, x: &Tensor, pos_offset: usize) -> Result<Tensor> {
        if x.rank() != 4 {
            return Err(Error::InvalidRank {
                context: "RoPECache::apply",
                rank: x.rank(),
                expected: 4,
            });
        }
        let batch = x.shape().dim(0)?;
        let heads = x.shape().dim(1)?;
        let seq_len = x.shape().dim(2)?;
        let hd = x.shape().dim(3)?;
        if hd != self.head_dim {
            return Err(Error::ShapeMismatch {
                context: "RoPECache::apply head_dim",
                expected: vec![batch, heads, seq_len, self.head_dim],
                actual: x.dims().to_vec(),
            });
        }
        let end_pos = pos_offset
            .checked_add(seq_len)
            .ok_or(Error::ArithmeticOverflow {
                context: "RoPECache::apply pos_offset",
            })?;
        if end_pos > self.max_seq_len {
            return Err(Error::IndexOutOfBounds {
                index: end_pos,
                len: self.max_seq_len,
            });
        }

        let src = x.to_f32_vec()?;
        let mut out = vec![0.0f32; src.len()];

        for b in 0..batch {
            for h in 0..heads {
                for t in 0..seq_len {
                    let pos = pos_offset + t;
                    let base = ((b * heads + h) * seq_len + t) * hd;
                    let freq_base = pos * self.half_dim;
                    for i in 0..self.half_dim {
                        let x0 = src.get(base + 2 * i).copied().unwrap_or(0.0);
                        let x1 = src.get(base + 2 * i + 1).copied().unwrap_or(0.0);
                        let c = self.cos.get(freq_base + i).copied().unwrap_or(1.0);
                        let s = self.sin.get(freq_base + i).copied().unwrap_or(0.0);
                        if let Some(o0) = out.get_mut(base + 2 * i) {
                            *o0 = x0 * c - x1 * s;
                        }
                        if let Some(o1) = out.get_mut(base + 2 * i + 1) {
                            *o1 = x0 * s + x1 * c;
                        }
                    }
                }
            }
        }

        Tensor::from_f32(&out, vec![batch, heads, seq_len, hd])
    }
}
