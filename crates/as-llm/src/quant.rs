//! Symmetric Int8 per-channel quantized linear projection (~4x weight memory reduction).

use as_core::{Error, F16, Result};
use as_tensor::Tensor;

/// Block size for 4-bit `Q4_0`-style quantization (32 weights per F16 scale).
pub const Q4_BLOCK_SIZE: usize = 32;

/// Int8 per-output-channel quantized weight matrix `[out_features, in_features]`.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    weights_i8: Vec<i8>,
    scales: Vec<f32>,
    out_features: usize,
    in_features: usize,
}

impl QuantizedLinear {
    /// Quantizes an `f32` weight tensor of shape `[out_features, in_features]` to Int8.
    pub fn from_f32_weight(weight: &Tensor) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(Error::InvalidRank {
                context: "QuantizedLinear::from_f32_weight",
                rank: weight.rank(),
                expected: 2,
            });
        }
        let out_features = weight.shape().dim(0)?;
        let in_features = weight.shape().dim(1)?;
        let data = weight.to_f32_vec()?;

        let mut weights_i8 = Vec::with_capacity(out_features * in_features);
        let mut scales = Vec::with_capacity(out_features);

        for row in 0..out_features {
            let row_slice = data
                .get(row * in_features..(row + 1) * in_features)
                .ok_or(Error::IndexOutOfBounds {
                    index: (row + 1) * in_features,
                    len: data.len(),
                })?;
            let max_abs = row_slice
                .iter()
                .fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales.push(scale);
            let inv_scale = 1.0 / scale;
            for &v in row_slice {
                let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                weights_i8.push(q);
            }
        }

        Ok(Self {
            weights_i8,
            scales,
            out_features,
            in_features,
        })
    }

    /// Returns weight storage size in bytes (excluding scale metadata).
    pub fn weight_bytes(&self) -> usize {
        self.weights_i8.len()
    }

    /// Dequantizes and executes `x @ W_q^T` for `x` of shape `[..., in_features]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut deq = Vec::with_capacity(self.out_features * self.in_features);
        for row in 0..self.out_features {
            let scale = self.scales.get(row).copied().unwrap_or(1.0);
            let start = row * self.in_features;
            let end = start + self.in_features;
            if let Some(slice) = self.weights_i8.get(start..end) {
                for &q in slice {
                    deq.push(f32::from(q) * scale);
                }
            }
        }
        let w = Tensor::from_f32(&deq, vec![self.out_features, self.in_features])?;
        let wt = w.transpose(0, 1)?;
        x.matmul(&wt)
    }
}

/// Block-wise 4-bit (`Q4_0`-style) quantized linear projection (~7.1x smaller than `f32`).
///
/// Packs 2 signed 4-bit weights (`-8..=7`) per byte with one [`F16`] scale per
/// [`Q4_BLOCK_SIZE`] (32) weights — 18 bytes per 32 weights (~4.5 bits/weight).
#[derive(Debug, Clone)]
pub struct Q4Linear {
    packed_nibbles: Vec<u8>,
    scales: Vec<F16>,
    out_features: usize,
    in_features: usize,
}

impl Q4Linear {
    /// Quantizes an `f32` weight tensor `[out_features, in_features]` into 4-bit blocks.
    pub fn from_f32_weight(weight: &Tensor) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(Error::InvalidRank {
                context: "Q4Linear::from_f32_weight",
                rank: weight.rank(),
                expected: 2,
            });
        }
        let out_features = weight.shape().dim(0)?;
        let in_features = weight.shape().dim(1)?;
        let data = weight.to_f32_vec()?;

        let total_elems = data.len();
        let num_blocks = total_elems.div_ceil(Q4_BLOCK_SIZE);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut packed_nibbles = Vec::with_capacity(total_elems.div_ceil(2));

        for block_idx in 0..num_blocks {
            let start = block_idx * Q4_BLOCK_SIZE;
            let end = (start + Q4_BLOCK_SIZE).min(total_elems);
            let slice = data.get(start..end).unwrap_or(&[]);

            let max_abs = slice.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale_f32 = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            let scale_f16 = F16::from_f32(scale_f32);
            let eff_scale = scale_f16.to_f32();
            let inv_scale = if eff_scale == 0.0 {
                0.0
            } else {
                1.0 / eff_scale
            };
            scales.push(scale_f16);

            let mut i = 0usize;
            while i < slice.len() {
                let v0 = slice.get(i).copied().unwrap_or(0.0);
                let q0 = ((v0 * inv_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                let q1 = if i + 1 < slice.len() {
                    let v1 = slice.get(i + 1).copied().unwrap_or(0.0);
                    ((v1 * inv_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8
                } else {
                    8u8 // zero nibble padding
                };
                packed_nibbles.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                i += 2;
            }
        }

        Ok(Self {
            packed_nibbles,
            scales,
            out_features,
            in_features,
        })
    }

    /// Total storage size in bytes (packed 4-bit weights + F16 block scales).
    pub fn storage_bytes(&self) -> usize {
        self.packed_nibbles.len() + self.scales.len() * 2
    }

    /// Compression ratio relative to uncompressed `f32` weights.
    pub fn compression_ratio(&self) -> f64 {
        let f32_bytes = (self.out_features * self.in_features * 4) as f64;
        f32_bytes / (self.storage_bytes().max(1) as f64)
    }

    /// Dequantizes 4-bit blocks and computes `x @ W_q4^T`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let total_elems = self.out_features * self.in_features;
        let num_blocks = total_elems.div_ceil(Q4_BLOCK_SIZE);
        let mut deq = Vec::with_capacity(total_elems);
        let mut byte_cursor = 0usize;

        for block_idx in 0..num_blocks {
            let block_len = (total_elems - block_idx * Q4_BLOCK_SIZE).min(Q4_BLOCK_SIZE);
            let scale = self
                .scales
                .get(block_idx)
                .map_or(1.0, |s| s.to_f32());
            let pairs = block_len.div_ceil(2);
            for p in 0..pairs {
                let packed = self.packed_nibbles.get(byte_cursor).copied().unwrap_or(0x88);
                byte_cursor += 1;
                let q0 = ((packed & 0x0F) as i8) - 8;
                deq.push(f32::from(q0) * scale);
                if p * 2 + 1 < block_len {
                    let q1 = (((packed >> 4) & 0x0F) as i8) - 8;
                    deq.push(f32::from(q1) * scale);
                }
            }
        }

        let w = Tensor::from_f32(&deq, vec![self.out_features, self.in_features])?;
        let wt = w.transpose(0, 1)?;
        x.matmul(&wt)
    }
}
