//! Safe, zero-dependency CSV dataset parser, writer, train/val splitter, and synthetic generator.

use as_core::{Error, Result};
use as_nn::Rng;

use crate::data::{Dataset, InMemoryDataset, Sample};

/// Maximum permitted CSV rows to guard against untrusted file DoS.
pub const MAX_CSV_ROWS: usize = 1_000_000;

/// Maximum permitted CSV columns per row.
pub const MAX_CSV_COLS: usize = 16_384;

/// Parses a CSV classification dataset where the last column is the integer class label (`0..C`)
/// and all preceding columns are finite floating-point features.
///
/// Returns `(dataset, num_classes)`.
pub fn parse_csv_dataset(csv_text: &str, has_header: bool) -> Result<(InMemoryDataset, usize)> {
    let mut samples = Vec::new();
    let mut max_label = 0usize;

    for (line_idx, raw_line) in csv_text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if has_header && line_idx == 0 {
            continue;
        }
        if samples.len() >= MAX_CSV_ROWS {
            return Err(Error::InvalidArgument {
                context: "parse_csv_dataset",
                reason: format!("CSV row count exceeds limit {MAX_CSV_ROWS}"),
            });
        }

        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() < 2 || cols.len() > MAX_CSV_COLS {
            return Err(Error::InvalidArgument {
                context: "parse_csv_dataset",
                reason: format!(
                    "line {} has {} columns (expected 2..={MAX_CSV_COLS})",
                    line_idx + 1,
                    cols.len()
                ),
            });
        }

        let feat_count = cols.len() - 1;
        let mut features = Vec::with_capacity(feat_count);
        for &c in cols.get(..feat_count).unwrap_or(&[]) {
            let v: f64 = c.parse().map_err(|_| Error::InvalidArgument {
                context: "parse_csv_dataset",
                reason: format!("line {}: invalid float feature `{c}`", line_idx + 1),
            })?;
            if !v.is_finite() {
                return Err(Error::InvalidArgument {
                    context: "parse_csv_dataset",
                    reason: format!("line {}: non-finite feature value", line_idx + 1),
                });
            }
            features.push(v);
        }

        let label_str = cols.last().copied().unwrap_or("");
        let label: usize = label_str.parse().map_err(|_| Error::InvalidArgument {
            context: "parse_csv_dataset",
            reason: format!("line {}: invalid integer label `{label_str}`", line_idx + 1),
        })?;
        if label > max_label {
            max_label = label;
        }
        samples.push(Sample { features, label });
    }

    if samples.is_empty() {
        return Err(Error::InvalidArgument {
            context: "parse_csv_dataset",
            reason: "CSV dataset contains zero samples".to_owned(),
        });
    }

    let ds = InMemoryDataset::new(samples)?;
    Ok((ds, max_label + 1))
}

/// Formats an [`InMemoryDataset`] as CSV text (last column = `label`).
pub fn format_csv_dataset(dataset: &InMemoryDataset, include_header: bool) -> Result<String> {
    let mut out = String::new();
    let feat_dim = dataset.feature_dim();
    if include_header {
        for i in 0..feat_dim {
            out.push_str(&format!("f{i},"));
        }
        out.push_str("label\n");
    }
    for idx in 0..dataset.len() {
        let s = dataset.get(idx)?;
        for &v in &s.features {
            out.push_str(&format!("{v:.6},"));
        }
        out.push_str(&format!("{}\n", s.label));
    }
    Ok(out)
}

/// Deterministically splits `dataset` into `(train_ds, val_ds)` using `val_ratio` in `(0.0, 1.0)`.
pub fn train_val_split(
    dataset: &InMemoryDataset,
    val_ratio: f64,
    rng: &mut Rng,
) -> Result<(InMemoryDataset, InMemoryDataset)> {
    if !(0.0..1.0).contains(&val_ratio) || val_ratio <= 0.0 {
        return Err(Error::InvalidArgument {
            context: "train_val_split",
            reason: "val_ratio must be strictly in (0.0, 1.0)".to_owned(),
        });
    }
    let n = dataset.len();
    if n < 2 {
        return Err(Error::InvalidArgument {
            context: "train_val_split",
            reason: "dataset must have at least 2 samples to split".to_owned(),
        });
    }
    let mut indices: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut indices);

    let val_len = (((n as f64) * val_ratio).round() as usize).clamp(1, n - 1);
    let mut val_samples = Vec::with_capacity(val_len);
    let mut train_samples = Vec::with_capacity(n - val_len);

    for (pos, &idx) in indices.iter().enumerate() {
        let s = dataset.get(idx)?;
        if pos < val_len {
            val_samples.push(s);
        } else {
            train_samples.push(s);
        }
    }
    Ok((
        InMemoryDataset::new(train_samples)?,
        InMemoryDataset::new(val_samples)?,
    ))
}

/// Generates a deterministic non-linear 3-class 2D synthetic classification dataset
/// (concentric rings + half-plane separation) for testing and demos.
pub fn generate_synthetic_classification_dataset(
    num_samples: usize,
    seed: u64,
) -> Result<InMemoryDataset> {
    if num_samples < 6 {
        return Err(Error::InvalidArgument {
            context: "generate_synthetic_classification_dataset",
            reason: "num_samples must be >= 6".to_owned(),
        });
    }
    let mut rng = Rng::from_seed(seed);
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let x = rng.uniform(-1.5, 1.5);
        let y = rng.uniform(-1.5, 1.5);
        let r2 = x * x + y * y;
        let label = if r2 < 0.55 {
            0
        } else if x >= 0.0 {
            1
        } else {
            2
        };
        samples.push(Sample {
            features: vec![x, y],
            label,
        });
    }
    InMemoryDataset::new(samples)
}
