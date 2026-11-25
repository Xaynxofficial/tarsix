//! Convolutional Neural Network (CNN) image classifier and 2D spatial pooling.

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, Result};
use as_nn::{Conv2d, Linear, Module, Param, Rng};
use as_tensor::Tensor;

/// 2D Max Pooling (`NCHW`) with square `kernel_size` and `stride`.
pub fn max_pool2d(x: &Tensor, kernel_size: usize, stride: usize) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(Error::InvalidRank {
            context: "max_pool2d",
            rank: x.rank(),
            expected: 4,
        });
    }
    if kernel_size == 0 || stride == 0 {
        return Err(Error::InvalidArgument {
            context: "max_pool2d",
            reason: "kernel_size and stride must be >= 1".to_owned(),
        });
    }
    let n = x.shape().dim(0)?;
    let c = x.shape().dim(1)?;
    let h = x.shape().dim(2)?;
    let w = x.shape().dim(3)?;
    if h < kernel_size || w < kernel_size {
        return Err(Error::InvalidArgument {
            context: "max_pool2d",
            reason: "spatial dimensions smaller than pooling kernel".to_owned(),
        });
    }
    let h_out = (h - kernel_size) / stride + 1;
    let w_out = (w - kernel_size) / stride + 1;
    let src = x.to_f32_vec()?;
    let mut out = vec![f32::NEG_INFINITY; n * c * h_out * w_out];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best = f32::NEG_INFINITY;
                    for kh in 0..kernel_size {
                        let ih = oh * stride + kh;
                        for kw in 0..kernel_size {
                            let iw = ow * stride + kw;
                            let idx = ((batch * c + ch) * h + ih) * w + iw;
                            let val = src.get(idx).copied().unwrap_or(f32::NEG_INFINITY);
                            if val > best {
                                best = val;
                            }
                        }
                    }
                    let dst_idx = ((batch * c + ch) * h_out + oh) * w_out + ow;
                    if let Some(slot) = out.get_mut(dst_idx) {
                        *slot = best;
                    }
                }
            }
        }
    }

    Tensor::from_f32(&out, vec![n, c, h_out, w_out])
}

/// Two-stage Convolutional Neural Network classifier with Global Average Pooling.
#[derive(Debug, Clone)]
pub struct CnnClassifier {
    /// First 3x3 convolution block.
    pub conv1: Conv2d,
    /// Second 3x3 convolution block.
    pub conv2: Conv2d,
    /// Classification head mapping pooled channels -> `num_classes`.
    pub head: Linear,
}

impl CnnClassifier {
    /// Constructs a deterministically initialized [`CnnClassifier`].
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        num_classes: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = Rng::from_seed(seed);
        let conv1 = Conv2d::new(
            "conv1",
            in_channels,
            hidden_channels,
            3,
            3,
            1,
            1,
            true,
            DType::F32,
            &mut rng,
        )?;
        let conv2 = Conv2d::new(
            "conv2",
            hidden_channels,
            hidden_channels * 2,
            3,
            3,
            2,
            1,
            true,
            DType::F32,
            &mut rng,
        )?;
        let head = Linear::new(
            "head",
            hidden_channels * 2,
            num_classes,
            true,
            DType::F32,
            &mut rng,
        )?;
        Ok(Self { conv1, conv2, head })
    }

    /// Forward pass returning `[N, num_classes]` logits and registered parameter [`VarId`]s.
    pub fn forward(&self, tape: &mut Tape, images: VarId) -> Result<(VarId, Vec<VarId>)> {
        let (h1, mut params) = self.conv1.forward(tape, images)?;
        let a1 = tape.relu(h1)?;
        let (h2, p2) = self.conv2.forward(tape, a1)?;
        params.extend(p2);
        let a2 = tape.relu(h2)?;
        // Global Average Pooling over H (axis 2) and W (axis 3): [N, C, H, W] -> [N, C]
        let gap_h = tape.mean_axis(a2, 3, false)?;
        let gap = tape.mean_axis(gap_h, 2, false)?;
        let (logits, p_head) = self.head.forward(tape, gap)?;
        params.extend(p_head);
        Ok((logits, params))
    }
}

impl Module for CnnClassifier {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = self.conv1.parameters_mut();
        p.extend(self.conv2.parameters_mut());
        p.extend(self.head.parameters_mut());
        p
    }
}
