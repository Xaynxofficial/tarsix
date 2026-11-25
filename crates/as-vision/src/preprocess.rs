//! Image buffer validation, bilinear resizing, CHW normalization, and batching.

use as_core::{Error, Result};
use as_tensor::Tensor;

/// Security ceiling for image width/height to prevent untrusted image OOM DoS.
pub const MAX_IMAGE_DIM: usize = 16_384;

/// Standard ImageNet per-channel RGB mean.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Standard ImageNet per-channel RGB standard deviation.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Validated interleaved `HWC` 8-bit image buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
    /// Number of channels (e.g., 1 for grayscale, 3 for RGB).
    pub channels: usize,
    /// Interleaved pixel bytes of length `height * width * channels`.
    pub data: Vec<u8>,
}

impl ImageBuffer {
    /// Creates a validated [`ImageBuffer`], rejecting oversized dimensions or mismatched lengths.
    pub fn new(height: usize, width: usize, channels: usize, data: Vec<u8>) -> Result<Self> {
        if height == 0 || width == 0 || channels == 0 {
            return Err(Error::InvalidArgument {
                context: "ImageBuffer::new",
                reason: "height, width, and channels must be >= 1".to_owned(),
            });
        }
        if height > MAX_IMAGE_DIM || width > MAX_IMAGE_DIM || channels > 4 {
            return Err(Error::InvalidArgument {
                context: "ImageBuffer::new",
                reason: format!(
                    "image dimensions ({height}x{width}x{channels}) exceed security limits"
                ),
            });
        }
        let expected = height
            .checked_mul(width)
            .and_then(|hw| hw.checked_mul(channels))
            .ok_or(Error::ArithmeticOverflow {
                context: "ImageBuffer::new",
            })?;
        if data.len() != expected {
            return Err(Error::BufferLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            height,
            width,
            channels,
            data,
        })
    }
}

/// Resizes `img` to `(target_h, target_w)` using bilinear interpolation.
pub fn resize_bilinear(
    img: &ImageBuffer,
    target_h: usize,
    target_w: usize,
) -> Result<ImageBuffer> {
    if target_h == 0 || target_w == 0 || target_h > MAX_IMAGE_DIM || target_w > MAX_IMAGE_DIM {
        return Err(Error::InvalidArgument {
            context: "resize_bilinear",
            reason: format!("invalid target size {target_h}x{target_w}"),
        });
    }
    let c = img.channels;
    let mut out = vec![0u8; target_h * target_w * c];

    let scale_y = (img.height as f64) / (target_h as f64);
    let scale_x = (img.width as f64) / (target_w as f64);

    for ty in 0..target_h {
        let src_y = ((ty as f64 + 0.5) * scale_y - 0.5).clamp(0.0, (img.height - 1) as f64);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(img.height - 1);
        let wy = src_y - (y0 as f64);

        for tx in 0..target_w {
            let src_x = ((tx as f64 + 0.5) * scale_x - 0.5).clamp(0.0, (img.width - 1) as f64);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(img.width - 1);
            let wx = src_x - (x0 as f64);

            for ch in 0..c {
                let p00 = f64::from(
                    img.data
                        .get((y0 * img.width + x0) * c + ch)
                        .copied()
                        .unwrap_or(0),
                );
                let p01 = f64::from(
                    img.data
                        .get((y0 * img.width + x1) * c + ch)
                        .copied()
                        .unwrap_or(0),
                );
                let p10 = f64::from(
                    img.data
                        .get((y1 * img.width + x0) * c + ch)
                        .copied()
                        .unwrap_or(0),
                );
                let p11 = f64::from(
                    img.data
                        .get((y1 * img.width + x1) * c + ch)
                        .copied()
                        .unwrap_or(0),
                );

                let top = p00 * (1.0 - wx) + p01 * wx;
                let bot = p10 * (1.0 - wx) + p11 * wx;
                let val = (top * (1.0 - wy) + bot * wy).round().clamp(0.0, 255.0) as u8;
                if let Some(slot) = out.get_mut((ty * target_w + tx) * c + ch) {
                    *slot = val;
                }
            }
        }
    }

    ImageBuffer::new(target_h, target_w, c, out)
}

/// Converts an [`ImageBuffer`] (`HWC`, `u8`) to a normalized `F32` tensor of shape `[1, C, H, W]`.
pub fn to_normalized_chw_tensor(
    img: &ImageBuffer,
    mean: &[f32],
    std: &[f32],
) -> Result<Tensor> {
    if mean.len() != img.channels || std.len() != img.channels {
        return Err(Error::BufferLengthMismatch {
            expected: img.channels,
            actual: mean.len(),
        });
    }
    let hw = img.height * img.width;
    let mut out = vec![0.0f32; img.channels * hw];

    for ch in 0..img.channels {
        let m = mean.get(ch).copied().unwrap_or(0.0);
        let s = std.get(ch).copied().unwrap_or(1.0);
        if s == 0.0 {
            return Err(Error::InvalidArgument {
                context: "to_normalized_chw_tensor",
                reason: "normalization std cannot be zero".to_owned(),
            });
        }
        let inv_s = 1.0 / s;
        for y in 0..img.height {
            for x in 0..img.width {
                let px = f32::from(
                    img.data
                        .get((y * img.width + x) * img.channels + ch)
                        .copied()
                        .unwrap_or(0),
                ) / 255.0;
                if let Some(slot) = out.get_mut(ch * hw + y * img.width + x) {
                    *slot = (px - m) * inv_s;
                }
            }
        }
    }

    Tensor::from_f32(&out, vec![1, img.channels, img.height, img.width])
}

/// Resizes, normalizes, and stacks a slice of images into a batch tensor `[N, C, target_h, target_w]`.
pub fn stack_batch(
    images: &[&ImageBuffer],
    target_h: usize,
    target_w: usize,
    mean: &[f32],
    std: &[f32],
) -> Result<Tensor> {
    if images.is_empty() {
        return Err(Error::InvalidArgument {
            context: "stack_batch",
            reason: "images slice cannot be empty".to_owned(),
        });
    }
    let channels = images.first().map_or(0, |im| im.channels);
    let mut batch_data = Vec::with_capacity(images.len() * channels * target_h * target_w);
    for img in images {
        if img.channels != channels {
            return Err(Error::InvalidArgument {
                context: "stack_batch",
                reason: "all images in batch must have identical channel counts".to_owned(),
            });
        }
        let resized = resize_bilinear(img, target_h, target_w)?;
        let t = to_normalized_chw_tensor(&resized, mean, std)?;
        batch_data.extend_from_slice(&t.to_f32_vec()?);
    }
    Tensor::from_f32(&batch_data, vec![images.len(), channels, target_h, target_w])
}
