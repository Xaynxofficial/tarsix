//! Bounded-memory Key-Value Cache for autoregressive Transformer decoding.

use as_core::{Error, Result};
use as_tensor::Tensor;

#[derive(Debug, Clone)]
struct LayerCache {
    // Stored as [batch, num_heads, cached_seq, head_dim]
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    seq_len: usize,
}

/// Per-layer Key-Value Cache enforcing `max_seq_len` memory ceilings.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl KvCache {
    /// Allocates an empty [`KvCache`] for `num_layers` decoder blocks.
    pub fn new(
        num_layers: usize,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerCache {
                k_data: Vec::new(),
                v_data: Vec::new(),
                seq_len: 0,
            })
            .collect();
        Self {
            layers,
            batch,
            num_heads,
            head_dim,
            max_seq_len,
        }
    }

    /// Returns the number of tokens currently cached in layer 0.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.seq_len)
    }

    /// Clears all cached keys and values without deallocating the outer structure.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.k_data.clear();
            layer.v_data.clear();
            layer.seq_len = 0;
        }
    }

    /// Appends `k_new` and `v_new` of shape `[batch, num_heads, new_seq, head_dim]` to `layer_idx`
    /// and returns the full concatenated `(K, V)` tensors of shape
    /// `[batch, num_heads, cached_seq + new_seq, head_dim]`.
    pub fn append(
        &mut self,
        layer_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let layer_count = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or(Error::IndexOutOfBounds {
                index: layer_idx,
                len: layer_count,
            })?;

        if k_new.rank() != 4 || v_new.rank() != 4 {
            return Err(Error::InvalidRank {
                context: "KvCache::append",
                rank: k_new.rank(),
                expected: 4,
            });
        }
        let b = k_new.shape().dim(0)?;
        let h = k_new.shape().dim(1)?;
        let new_seq = k_new.shape().dim(2)?;
        let hd = k_new.shape().dim(3)?;
        if b != self.batch || h != self.num_heads || hd != self.head_dim {
            return Err(Error::ShapeMismatch {
                context: "KvCache::append",
                expected: vec![self.batch, self.num_heads, new_seq, self.head_dim],
                actual: k_new.dims().to_vec(),
            });
        }

        let total_seq = layer
            .seq_len
            .checked_add(new_seq)
            .ok_or(Error::ArithmeticOverflow {
                context: "KvCache::append seq_len",
            })?;
        if total_seq > self.max_seq_len {
            return Err(Error::InvalidArgument {
                context: "KvCache::append",
                reason: format!(
                    "sequence length {total_seq} exceeds KV-Cache limit {}",
                    self.max_seq_len
                ),
            });
        }

        let k_src = k_new.to_f32_vec()?;
        let v_src = v_new.to_f32_vec()?;

        let old_seq = layer.seq_len;
        let bh = b * h;
        let mut next_k = Vec::with_capacity(bh * total_seq * hd);
        let mut next_v = Vec::with_capacity(bh * total_seq * hd);

        for bh_idx in 0..bh {
            let old_start = bh_idx * old_seq * hd;
            let old_end = old_start + old_seq * hd;
            if let Some(slice) = layer.k_data.get(old_start..old_end) {
                next_k.extend_from_slice(slice);
            }
            if let Some(slice) = layer.v_data.get(old_start..old_end) {
                next_v.extend_from_slice(slice);
            }

            let new_start = bh_idx * new_seq * hd;
            let new_end = new_start + new_seq * hd;
            if let Some(slice) = k_src.get(new_start..new_end) {
                next_k.extend_from_slice(slice);
            }
            if let Some(slice) = v_src.get(new_start..new_end) {
                next_v.extend_from_slice(slice);
            }
        }

        layer.k_data = next_k;
        layer.v_data = next_v;
        layer.seq_len = total_seq;

        let k_full = Tensor::from_f32(&layer.k_data, vec![b, h, total_seq, hd])?;
        let v_full = Tensor::from_f32(&layer.v_data, vec![b, h, total_seq, hd])?;
        Ok((k_full, v_full))
    }
}
