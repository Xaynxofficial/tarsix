//! Trainable YOLO-style Object Detection (`BBox`, `iou`, `nms`, `ObjectDetector`, `detection_loss`).
//!
//! Uses normalized `[0, 1]` center-format bounding boxes `(cx, cy, w, h)` and a
//! fully differentiable single-stage grid head (`[N, 5 + num_classes, S, S]`) so users
//! can both **train from scratch / fine-tune** on custom bounding boxes and run NMS inference.

use std::fmt::Write as _;

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, Result};
use as_nn::{Conv2d, Module, Param, Rng, mse_loss};
use as_tensor::Tensor;

use crate::preprocess::ImageBuffer;

/// Normalized `[0.0, 1.0]` center-format bounding box `(cx, cy, w, h)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Center X in `[0, 1]`.
    pub cx: f32,
    /// Center Y in `[0, 1]`.
    pub cy: f32,
    /// Box width in `[0, 1]`.
    pub w: f32,
    /// Box height in `[0, 1]`.
    pub h: f32,
}

impl BBox {
    /// Converts `(cx, cy, w, h)` to corner coordinates `(x1, y1, x2, y2)`.
    pub fn to_xyxy(self) -> (f32, f32, f32, f32) {
        let half_w = self.w * 0.5;
        let half_h = self.h * 0.5;
        (
            self.cx - half_w,
            self.cy - half_h,
            self.cx + half_w,
            self.cy + half_h,
        )
    }
}

/// Ground-truth labeled bounding box supplied by users for training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabeledBox {
    /// Target bounding box in normalized `[0, 1]` coordinates.
    pub bbox: BBox,
    /// Target integer class index `0..num_classes`.
    pub class_id: usize,
}

/// Predicted object detection returned by [`decode_detections`] and [`nms`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Predicted normalized bounding box.
    pub bbox: BBox,
    /// Objectness * class confidence score.
    pub confidence: f32,
    /// Predicted class index.
    pub class_id: usize,
}

/// Computes Intersection over Union (`IoU`) between two [`BBox`]es in `[0.0, 1.0]`.
pub fn iou(a: BBox, b: BBox) -> f32 {
    let (ax1, ay1, ax2, ay2) = a.to_xyxy();
    let (bx1, by1, bx2, by2) = b.to_xyxy();

    let inter_w = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let inter_h = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 1e-12 {
        0.0
    } else {
        inter / union
    }
}

/// Class-aware Non-Maximum Suppression (NMS), discarding overlapping boxes with `IoU > iou_threshold`.
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    let mut sorted = detections.to_vec();
    sorted.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut kept: Vec<Detection> = Vec::new();
    for cand in sorted {
        let suppressed = kept
            .iter()
            .any(|k| k.class_id == cand.class_id && iou(k.bbox, cand.bbox) > iou_threshold);
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// Encodes per-image ground-truth [`LabeledBox`] lists into a YOLO target grid tensor
/// of shape `[batch, 5 + num_classes, grid_h, grid_w]`.
pub fn encode_yolo_targets(
    batch_boxes: &[Vec<LabeledBox>],
    grid_h: usize,
    grid_w: usize,
    num_classes: usize,
) -> Result<Tensor> {
    let batch = batch_boxes.len();
    let channels = 5 + num_classes;
    let mut grid = vec![0.0f32; batch * channels * grid_h * grid_w];
    let hw = grid_h * grid_w;

    for (b, boxes) in batch_boxes.iter().enumerate() {
        for lb in boxes {
            if lb.class_id >= num_classes {
                return Err(Error::IndexOutOfBounds {
                    index: lb.class_id,
                    len: num_classes,
                });
            }
            let cx = lb.bbox.cx.clamp(0.0, 0.9999);
            let cy = lb.bbox.cy.clamp(0.0, 0.9999);
            let gx = ((cx * (grid_w as f32)).floor() as usize).min(grid_w - 1);
            let gy = ((cy * (grid_h as f32)).floor() as usize).min(grid_h - 1);

            let dx = cx * (grid_w as f32) - (gx as f32);
            let dy = cy * (grid_h as f32) - (gy as f32);

            let set = |g: &mut [f32], ch: usize, val: f32| {
                let idx = (b * channels + ch) * hw + gy * grid_w + gx;
                if let Some(slot) = g.get_mut(idx) {
                    *slot = val;
                }
            };
            set(&mut grid, 0, dx);
            set(&mut grid, 1, dy);
            set(&mut grid, 2, lb.bbox.w.clamp(0.0, 1.0));
            set(&mut grid, 3, lb.bbox.h.clamp(0.0, 1.0));
            set(&mut grid, 4, 1.0); // objectness = 1
            set(&mut grid, 5 + lb.class_id, 1.0);
        }
    }

    Tensor::from_f32(&grid, vec![batch, channels, grid_h, grid_w])
}

/// Decodes raw YOLO grid predictions of shape `[1, 5 + num_classes, grid_h, grid_w]`
/// into filtered, NMS-suppressed [`Detection`]s.
pub fn decode_detections(
    pred: &Tensor,
    num_classes: usize,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    if pred.rank() != 4 {
        return Err(Error::InvalidRank {
            context: "decode_detections",
            rank: pred.rank(),
            expected: 4,
        });
    }
    let channels = pred.shape().dim(1)?;
    let grid_h = pred.shape().dim(2)?;
    let grid_w = pred.shape().dim(3)?;
    if channels != 5 + num_classes {
        return Err(Error::ShapeMismatch {
            context: "decode_detections channels",
            expected: vec![1, 5 + num_classes, grid_h, grid_w],
            actual: pred.dims().to_vec(),
        });
    }

    let data = pred.to_f32_vec()?;
    let hw = grid_h * grid_w;
    let get = |ch: usize, gy: usize, gx: usize| -> f32 {
        data.get(ch * hw + gy * grid_w + gx).copied().unwrap_or(0.0)
    };

    let mut raw = Vec::new();
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let obj = get(4, gy, gx);
            if obj < conf_threshold {
                continue;
            }
            let mut best_cls = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let s = get(5 + c, gy, gx);
                if s > best_score {
                    best_score = s;
                    best_cls = c;
                }
            }
            let conf = obj * best_score.clamp(0.0, 1.0);
            if conf < conf_threshold {
                continue;
            }
            let cx = ((gx as f32) + get(0, gy, gx).clamp(0.0, 1.0)) / (grid_w as f32);
            let cy = ((gy as f32) + get(1, gy, gx).clamp(0.0, 1.0)) / (grid_h as f32);
            let w = get(2, gy, gx).clamp(0.0, 1.0);
            let h = get(3, gy, gx).clamp(0.0, 1.0);
            raw.push(Detection {
                bbox: BBox { cx, cy, w, h },
                confidence: conf,
                class_id: best_cls,
            });
        }
    }

    Ok(nms(&raw, iou_threshold))
}

/// Differentiable YOLO grid detection loss (coordinates + objectness + class scores).
pub fn detection_loss(tape: &mut Tape, pred_grid: VarId, target_grid: VarId) -> Result<VarId> {
    mse_loss(tape, pred_grid, target_grid)
}

/// End-to-end trainable single-stage YOLO-style Object Detector.
///
/// Downsamples input `[N, C, H, W]` by 4x (`H/4 x W/4` grid) and predicts
/// `[N, 5 + num_classes, H/4, W/4]`.
#[derive(Debug, Clone)]
pub struct ObjectDetector {
    /// Stride-2 stem convolution (`H/2 x W/2`).
    pub stem: Conv2d,
    /// Stride-2 backbone convolution (`H/4 x W/4`).
    pub backbone: Conv2d,
    /// 1x1 YOLO grid detection head (`5 + num_classes` output channels).
    pub head: Conv2d,
    num_classes: usize,
}

impl ObjectDetector {
    /// Creates a deterministically initialized [`ObjectDetector`].
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        num_classes: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = Rng::from_seed(seed);
        let stem = Conv2d::new(
            "det.stem",
            in_channels,
            hidden_channels,
            3,
            3,
            2,
            1,
            true,
            DType::F32,
            &mut rng,
        )?;
        let backbone = Conv2d::new(
            "det.backbone",
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
        let head = Conv2d::new(
            "det.head",
            hidden_channels * 2,
            5 + num_classes,
            1,
            1,
            1,
            0,
            true,
            DType::F32,
            &mut rng,
        )?;
        Ok(Self {
            stem,
            backbone,
            head,
            num_classes,
        })
    }

    /// Returns the number of object classes supported by this detector.
    pub const fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Differentiable forward pass returning raw grid `[N, 5 + num_classes, H/4, W/4]`
    /// and registered parameter [`VarId`]s for backpropagation.
    pub fn forward(&self, tape: &mut Tape, images: VarId) -> Result<(VarId, Vec<VarId>)> {
        let (h1, mut params) = self.stem.forward(tape, images)?;
        let a1 = tape.relu(h1)?;
        let (h2, p2) = self.backbone.forward(tape, a1)?;
        params.extend(p2);
        let a2 = tape.relu(h2)?;
        let (grid, p3) = self.head.forward(tape, a2)?;
        params.extend(p3);
        Ok((grid, params))
    }
}

impl Module for ObjectDetector {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = self.stem.parameters();
        p.extend(self.backbone.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = self.stem.parameters_mut();
        p.extend(self.backbone.parameters_mut());
        p.extend(self.head.parameters_mut());
        p
    }
}

/// Renders an [`ImageBuffer`] with overlaid YOLO [`Detection`] bounding boxes and labels
/// as a crisp, minimal white-framed SVG (`400x440`).
pub fn render_detections_svg(img: &ImageBuffer, detections: &[Detection]) -> String {
    let canvas_w = 400.0f32;
    let canvas_h = 440.0f32;
    let pad = 24.0f32;
    let img_box = 352.0f32;
    let cell_w = img_box / (img.width.max(1) as f32);
    let cell_h = img_box / (img.height.max(1) as f32);

    let mut svg = String::with_capacity(8192);
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}" width="{canvas_w}" height="{canvas_h}">
  <rect x="0.5" y="0.5" width="{bw}" height="{bh}" rx="12" fill="#ffffff" stroke="#e5e7eb"/>
  <text x="24" y="34" fill="#111827" font-family="-apple-system,BlinkMacSystemFont,Inter,sans-serif" font-size="15" font-weight="600">Tarsix · YOLO Detection Output</text>
  <text x="24" y="52" fill="#6b7280" font-family="-apple-system,BlinkMacSystemFont,Inter,sans-serif" font-size="11.5">{w}x{h} RGB · {n_det} object(s) detected</text>
  <g transform="translate({pad}, 66)">
"##,
        bw = canvas_w - 1.0,
        bh = canvas_h - 1.0,
        w = img.width,
        h = img.height,
        n_det = detections.len(),
    );

    // Draw image pixels
    for y in 0..img.height {
        for x in 0..img.width {
            let base = (y * img.width + x) * img.channels;
            let r = img.data.get(base).copied().unwrap_or(0);
            let g = img.data.get(base + 1).copied().unwrap_or(r);
            let b = img.data.get(base + 2).copied().unwrap_or(r);
            let px = (x as f32) * cell_w;
            let py = (y as f32) * cell_h;
            let _ = write!(
                svg,
                "    <rect x=\"{px:.1}\" y=\"{py:.1}\" width=\"{cw:.1}\" height=\"{ch:.1}\" fill=\"rgb({r},{g},{b})\"/>\n",
                cw = cell_w + 0.4,
                ch = cell_h + 0.4,
            );
        }
    }

    // Draw predicted YOLO bounding boxes + confidence badges
    let palette = ["#10b981", "#2563eb", "#f59e0b", "#ef4444"];
    for d in detections {
        let (x1, y1, x2, y2) = d.bbox.to_xyxy();
        let bx = (x1.clamp(0.0, 1.0)) * img_box;
        let by = (y1.clamp(0.0, 1.0)) * img_box;
        let bw = ((x2 - x1).clamp(0.0, 1.0)) * img_box;
        let bh = ((y2 - y1).clamp(0.0, 1.0)) * img_box;
        let color = palette.get(d.class_id % palette.len()).copied().unwrap_or("#10b981");
        let pct = (d.confidence.clamp(0.0, 1.0)) * 100.0;
        let lbl_y = (by - 20.0).max(0.0);
        let _ = write!(
            svg,
            r##"    <rect x="{bx:.1}" y="{by:.1}" width="{bw:.1}" height="{bh:.1}" fill="none" stroke="{color}" stroke-width="2.5" rx="3"/>
    <rect x="{bx:.1}" y="{lbl_y:.1}" width="108" height="18" rx="3" fill="{color}"/>
    <text x="{tx:.1}" y="{ty:.1}" fill="#ffffff" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10.5" font-weight="600">cls:{cls} {pct:.0}%</text>
"##,
            tx = bx + 6.0,
            ty = lbl_y + 12.5,
            cls = d.class_id,
        );
    }

    svg.push_str("  </g>\n</svg>\n");
    svg
}
