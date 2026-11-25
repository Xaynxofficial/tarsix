//! Zero-dependency, minimal light-mode SVG training report generator.
//!
//! Renders clean, publication-ready **Training Loss** and **Validation Accuracy (%)**
//! curves directly from [`crate::EpochMetrics`] without external plotting libraries.

use std::fmt::Write as _;

use as_core::{Error, Result};

use crate::metrics::EpochMetrics;

/// Renders a crisp, minimal white `960x400` SVG report showing Training Loss (left)
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

    let left_x = 68.0f64;
    let right_x = 532.0f64;
    let plot_y = 112.0f64;
    let plot_w = 388.0f64;
    let plot_h = 228.0f64;

    let mut loss_pts = String::new();
    let mut loss_area = String::new();
    let mut acc_pts = String::new();
    let mut acc_area = String::new();
    let mut last_loss_xy = (left_x, plot_y + plot_h);
    let mut last_acc_xy = (right_x, plot_y + plot_h);
    let mut acc_count = 0usize;

    let denom_x = (epochs.saturating_sub(1)).max(1) as f64;
    for (i, m) in history.iter().enumerate() {
        let t = (i as f64) / denom_x;
        let lx = left_x + t * plot_w;
        let ly = plot_y + plot_h - ((m.train_loss - min_loss) / loss_span) * plot_h;
        last_loss_xy = (lx, ly);
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
            last_acc_xy = (ax, ay);
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
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 400" width="960" height="400">
  <defs>
    <linearGradient id="lossWash" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#111827" stop-opacity="0.08"/>
      <stop offset="100%" stop-color="#111827" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="accWash" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#059669" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#059669" stop-opacity="0.0"/>
    </linearGradient>
  </defs>

  <rect x="0.5" y="0.5" width="959" height="399" rx="12" fill="#ffffff" stroke="#e5e7eb"/>

  <!-- Header -->
  <text x="36" y="42" fill="#111827" font-family="-apple-system,BlinkMacSystemFont,Inter,Segoe UI,sans-serif" font-size="18" font-weight="600" letter-spacing="-0.02em">Tarsix · {safe_title}</text>
  <text x="36" y="64" fill="#6b7280" font-family="-apple-system,BlinkMacSystemFont,Inter,Segoe UI,sans-serif" font-size="12.5">Epochs {epochs}   ·   Final Loss {final_loss:.4}   ·   Best Val Accuracy {best_pct:.1}%</text>
  <line x1="36" y1="80" x2="924" y2="80" stroke="#f3f4f6"/>

  <!-- Left Panel: Training Loss -->
  <text x="{left_x}" y="102" fill="#374151" font-family="-apple-system,BlinkMacSystemFont,Inter,Segoe UI,sans-serif" font-size="12" font-weight="600">Training Loss</text>
  <line x1="{left_x}" y1="{plot_y}" x2="{lx_end}" y2="{plot_y}" stroke="#f3f4f6"/>
  <line x1="{left_x}" y1="{mid_y}" x2="{lx_end}" y2="{mid_y}" stroke="#f3f4f6"/>
  <line x1="{left_x}" y1="{bot_y}" x2="{lx_end}" y2="{bot_y}" stroke="#e5e7eb"/>
  <text x="58" y="{y_top_lbl}" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">{max_loss:.2}</text>
  <text x="58" y="{y_mid_lbl}" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">{mid_loss:.2}</text>
  <text x="58" y="{y_bot_lbl}" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">{min_loss:.2}</text>
  <text x="{left_x}" y="362" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">1</text>
  <text x="{lx_end}" y="362" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">{epochs}</text>
  <path d="{loss_area}" fill="url(#lossWash)"/>
  <polyline fill="none" stroke="#111827" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{loss_pts}"/>
  <circle cx="{llx:.1}" cy="{lly:.1}" r="3.2" fill="#111827"/>

  <!-- Right Panel: Validation Accuracy -->
  <text x="{right_x}" y="102" fill="#374151" font-family="-apple-system,BlinkMacSystemFont,Inter,Segoe UI,sans-serif" font-size="12" font-weight="600">Validation Accuracy</text>
  <line x1="{right_x}" y1="{plot_y}" x2="{rx_end}" y2="{plot_y}" stroke="#f3f4f6"/>
  <line x1="{right_x}" y1="{mid_y}" x2="{rx_end}" y2="{mid_y}" stroke="#f3f4f6"/>
  <line x1="{right_x}" y1="{bot_y}" x2="{rx_end}" y2="{bot_y}" stroke="#e5e7eb"/>
  <text x="522" y="{y_top_lbl}" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">100%</text>
  <text x="522" y="{y_mid_lbl}" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">50%</text>
  <text x="522" y="{y_bot_lbl}" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">0%</text>
  <text x="{right_x}" y="362" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">1</text>
  <text x="{rx_end}" y="362" text-anchor="end" fill="#9ca3af" font-family="ui-monospace,SFMono-Regular,monospace" font-size="10">{epochs}</text>
  <path d="{acc_area}" fill="url(#accWash)"/>
  <polyline fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{acc_pts}"/>
  <circle cx="{lax:.1}" cy="{lay:.1}" r="3.2" fill="#059669"/>
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
        llx = last_loss_xy.0,
        lly = last_loss_xy.1,
        lax = last_acc_xy.0,
        lay = last_acc_xy.1,
    );

    Ok(svg)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
