//! Zero-dependency, publication-quality SVG training report generator.
//!
//! Renders dual-panel **Training Loss** and **Validation Accuracy (%)** curves
//! directly from [`crate::EpochMetrics`] without any external plotting libraries.

use std::fmt::Write as _;

use as_core::{Error, Result};

use crate::metrics::EpochMetrics;

/// Renders a crisp, dark-themed `960x420` SVG report showing Training Loss (left)
/// and Validation Accuracy (right) over epochs.
pub fn render_training_svg(title: &str, history: &[EpochMetrics]) -> Result<String> {
    if history.is_empty() {
        return Err(Error::InvalidArgument {
            context: "render_training_svg",
            reason: "cannot render training chart from empty metrics history".to_owned(),
        });
    }

    let safe_title = escape_xml(title);
    let epochs = history.len();
    let final_loss = history.last().map_or(0.0, |m| m.train_loss);
    let best_acc = history
        .iter()
        .filter_map(|m| m.val_accuracy)
        .fold(0.0f64, f64::max);

    let max_loss = history
        .iter()
        .map(|m| m.train_loss)
        .fold(0.0f64, f64::max)
        .max(1e-6);
    let min_loss = history
        .iter()
        .map(|m| m.train_loss)
        .fold(f64::INFINITY, f64::min)
        .min(0.0);
    let loss_span = (max_loss - min_loss).max(1e-6);

    // Panel geometry
    let left_x = 64.0f64;
    let right_x = 524.0f64;
    let plot_y = 112.0f64;
    let plot_w = 396.0f64;
    let plot_h = 248.0f64;

    let mut loss_pts = String::new();
    let mut loss_area = String::new();
    let mut acc_pts = String::new();
    let mut acc_area = String::new();
    let mut acc_count = 0usize;

    let denom_x = (epochs.saturating_sub(1)).max(1) as f64;
    for (i, m) in history.iter().enumerate() {
        let t = (i as f64) / denom_x;
        let lx = left_x + t * plot_w;
        let ly = plot_y + plot_h - ((m.train_loss - min_loss) / loss_span) * plot_h;
        let _ = write!(loss_pts, "{lx:.1},{ly:.1} ");
        if i == 0 {
            let _ = write!(loss_area, "M {lx:.1},{:.1} L {lx:.1},{ly:.1}", plot_y + plot_h);
        } else {
            let _ = write!(loss_area, " L {lx:.1},{ly:.1}");
        }

        if let Some(acc) = m.val_accuracy {
            let ax = right_x + t * plot_w;
            let clamped = acc.clamp(0.0, 1.0);
            let ay = plot_y + plot_h - clamped * plot_h;
            let _ = write!(acc_pts, "{ax:.1},{ay:.1} ");
            if acc_count == 0 {
                let _ = write!(acc_area, "M {ax:.1},{:.1} L {ax:.1},{ay:.1}", plot_y + plot_h);
            } else {
                let _ = write!(acc_area, " L {ax:.1},{ay:.1}");
            }
            acc_count += 1;
        }
    }
    let _ = write!(
        loss_area,
        " L {:.1},{:.1} Z",
        left_x + plot_w,
        plot_y + plot_h
    );
    if acc_count > 0 {
        let _ = write!(
            acc_area,
            " L {:.1},{:.1} Z",
            right_x + plot_w,
            plot_y + plot_h
        );
    }

    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 420" width="960" height="420">
  <defs>
    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.32"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#10b981" stop-opacity="0.32"/>
      <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
    </linearGradient>
  </defs>
  <rect width="960" height="420" rx="14" fill="#0f172a"/>
  <text x="32" y="40" fill="#f8fafc" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="20" font-weight="700">TARSIX — {safe_title}</text>
  <text x="32" y="64" fill="#94a3b8" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="13">Epochs: {epochs}   ·   Final Loss: {final_loss:.4}   ·   Best Val Accuracy: {best_pct:.1}%</text>

  <!-- Left Card: Training Loss -->
  <rect x="24" y="82" width="448" height="314" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1"/>
  <text x="44" y="104" fill="#e2e8f0" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="13" font-weight="600">Training Loss</text>
  <line x1="{left_x}" y1="{plot_y}" x2="{lx_end}" y2="{plot_y}" stroke="#334155" stroke-dasharray="3,4"/>
  <line x1="{left_x}" y1="{mid_y}" x2="{lx_end}" y2="{mid_y}" stroke="#334155" stroke-dasharray="3,4"/>
  <line x1="{left_x}" y1="{bot_y}" x2="{lx_end}" y2="{bot_y}" stroke="#475569"/>
  <text x="56" y="{y_top_lbl}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">{max_loss:.2}</text>
  <text x="56" y="{y_mid_lbl}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">{mid_loss:.2}</text>
  <text x="56" y="{y_bot_lbl}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">{min_loss:.2}</text>
  <text x="{left_x}" y="378" fill="#94a3b8" font-family="monospace" font-size="10">1</text>
  <text x="{lx_end}" y="378" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">{epochs}</text>
  <path d="{loss_area}" fill="url(#lossGrad)"/>
  <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linejoin="round" points="{loss_pts}"/>

  <!-- Right Card: Validation Accuracy -->
  <rect x="488" y="82" width="448" height="314" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1"/>
  <text x="508" y="104" fill="#e2e8f0" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="13" font-weight="600">Validation Accuracy (%)</text>
  <line x1="{right_x}" y1="{plot_y}" x2="{rx_end}" y2="{plot_y}" stroke="#334155" stroke-dasharray="3,4"/>
  <line x1="{right_x}" y1="{mid_y}" x2="{rx_end}" y2="{mid_y}" stroke="#334155" stroke-dasharray="3,4"/>
  <line x1="{right_x}" y1="{bot_y}" x2="{rx_end}" y2="{bot_y}" stroke="#475569"/>
  <text x="516" y="{y_top_lbl}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">100%</text>
  <text x="516" y="{y_mid_lbl}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">50%</text>
  <text x="516" y="{y_bot_lbl}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">0%</text>
  <text x="{right_x}" y="378" fill="#94a3b8" font-family="monospace" font-size="10">1</text>
  <text x="{rx_end}" y="378" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="10">{epochs}</text>
  <path d="{acc_area}" fill="url(#accGrad)"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" points="{acc_pts}"/>
</svg>
"##,
        best_pct = best_acc * 100.0,
        lx_end = left_x + plot_w,
        rx_end = right_x + plot_w,
        mid_y = plot_y + plot_h * 0.5,
        bot_y = plot_y + plot_h,
        y_top_lbl = plot_y + 4.0,
        y_mid_lbl = plot_y + plot_h * 0.5 + 4.0,
        y_bot_lbl = plot_y + plot_h + 4.0,
        mid_loss = (max_loss + min_loss) * 0.5,
    );

    Ok(svg)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
