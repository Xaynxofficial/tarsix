//! Symmetric Int8 per-channel quantized linear projection (~4x weight memory reduction).

use as_core::{Error, Result};
use as_tensor::Tensor;

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
