//! Dataset abstraction and deterministic seeded [`DataLoader`].

use as_core::{DType, Error, Result};
use as_nn::Rng;
use as_tensor::Tensor;

/// A single training/evaluation sample pairing feature values and a target class or vector.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Flattened feature vector of shape `[feature_dim]`.
    pub features: Vec<f64>,
    /// Optional class label (for classification).
    pub label: usize,
}

/// Stacked mini-batch yielded by [`DataLoader`].
#[derive(Debug, Clone)]
pub struct Batch {
    /// Feature tensor of shape `[batch_size, feature_dim]`.
    pub inputs: Tensor,
    /// Target class indices of length `batch_size`.
    pub labels: Vec<usize>,
}

/// Abstract dataset contract.
pub trait Dataset {
    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns `true` if the dataset contains zero samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Feature dimensionality per sample.
    fn feature_dim(&self) -> usize;

    /// Retrieves sample at `index`.
    fn get(&self, index: usize) -> Result<Sample>;
}

/// Simple in-memory dataset for tabular, synthetic, or pre-loaded feature vectors.
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    samples: Vec<Sample>,
    feature_dim: usize,
}

impl InMemoryDataset {
    /// Creates a validated [`InMemoryDataset`], verifying uniform feature dimensions.
    pub fn new(samples: Vec<Sample>) -> Result<Self> {
        let feature_dim = samples.first().map_or(0, |s| s.features.len());
        for s in &samples {
            if s.features.len() != feature_dim {
                return Err(Error::BufferLengthMismatch {
                    expected: feature_dim,
                    actual: s.features.len(),
                });
            }
        }
        Ok(Self {
            samples,
            feature_dim,
        })
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn get(&self, index: usize) -> Result<Sample> {
        self.samples
            .get(index)
            .cloned()
            .ok_or(Error::IndexOutOfBounds {
                index,
                len: self.samples.len(),
            })
    }
}

/// Deterministic mini-batch loader with optional seeded shuffling.
#[derive(Debug, Clone)]
pub struct DataLoader {
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    dtype: DType,
}

impl DataLoader {
    /// Creates a new [`DataLoader`].
    pub fn new(batch_size: usize, shuffle: bool, drop_last: bool, dtype: DType) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidArgument {
                context: "DataLoader::new",
                reason: "batch_size must be >= 1".to_owned(),
            });
        }
        Ok(Self {
            batch_size,
            shuffle,
            drop_last,
            dtype,
        })
    }

    /// Materializes all mini-batches for one epoch using deterministic `rng`.
    pub fn epoch_batches<D: Dataset>(&self, dataset: &D, rng: &mut Rng) -> Result<Vec<Batch>> {
        let n = dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();
        if self.shuffle {
            rng.shuffle(&mut indices);
        }

        let feat_dim = dataset.feature_dim();
        let mut batches = Vec::new();
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + self.batch_size).min(n);
            let cur_bs = end - pos;
            if self.drop_last && cur_bs < self.batch_size {
                break;
            }
            let mut flat = Vec::with_capacity(cur_bs * feat_dim);
            let mut labels = Vec::with_capacity(cur_bs);
            for &idx in indices.get(pos..end).unwrap_or(&[]) {
                let sample = dataset.get(idx)?;
                flat.extend_from_slice(&sample.features);
                labels.push(sample.label);
            }
            let t_f64 = Tensor::from_f64(&flat, vec![cur_bs, feat_dim])?;
            let inputs = if self.dtype == DType::F64 {
                t_f64
            } else {
                t_f64.to_dtype(self.dtype)?
            };
            batches.push(Batch { inputs, labels });
            pos = end;
        }
        Ok(batches)
    }
}
