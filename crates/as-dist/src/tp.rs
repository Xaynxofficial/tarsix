//! Megatron-LM style **Tensor Parallelism** (`ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelMlp`).
//!
//! Splits large Transformer MLP and Attention projection matrices across `world_size` workers:
//! - [`ColumnParallelLinear`]: shards output features `[out_features / P, in_features]` (zero comm in forward)
//! - [`RowParallelLinear`]: shards input features `[out_features, in_features / P]` + single `AllReduce(Sum)`
//! - [`TensorParallelMlp`]: full `ColumnParallel -> ReLU -> RowParallel` block requiring only **1 `AllReduce`**

use as_core::{DType, Error, Result};
use as_tensor::Tensor;

use crate::comm::{AllReduceOp, Communicator};

/// Column-parallel linear projection sharded along the output dimension (`out_features / world_size`).
#[derive(Debug, Clone)]
pub struct ColumnParallelLinear {
    /// Local weight shard `[out_shard, in_features]`.
    pub weight_shard: Tensor,
    /// Optional local bias shard `[1, out_shard]`.
    pub bias_shard: Option<Tensor>,
    /// Whether to `AllGather` outputs across ranks (false when chained into [`RowParallelLinear`]).
    pub gather_output: bool,
}

impl ColumnParallelLinear {
    /// Shards a full `[out_features, in_features]` weight (and optional `[1, out_features]` bias)
    /// for `rank` out of `world_size`.
    pub fn from_full_weights(
        full_weight: &Tensor,
        full_bias: Option<&Tensor>,
        rank: usize,
        world_size: usize,
        gather_output: bool,
    ) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(Error::InvalidArgument {
                context: "ColumnParallelLinear::from_full_weights",
                reason: format!("invalid rank {rank} for world_size {world_size}"),
            });
        }
        let out_dim = full_weight.shape().dim(0)?;
        let in_dim = full_weight.shape().dim(1)?;
        if out_dim % world_size != 0 {
            return Err(Error::InvalidArgument {
                context: "ColumnParallelLinear::from_full_weights",
                reason: format!("out_features ({out_dim}) must be divisible by world_size ({world_size})"),
            });
        }
        let shard_out = out_dim / world_size;
        let row_start = rank * shard_out;
        let full_vals = full_weight.to_dtype(DType::F32)?.to_f32_vec()?;
        let start = row_start * in_dim;
        let end = (row_start + shard_out) * in_dim;
        let shard_slice = full_vals.get(start..end).ok_or(Error::IndexOutOfBounds {
            index: end,
            len: full_vals.len(),
        })?;
        let weight_shard = Tensor::from_f32(shard_slice, vec![shard_out, in_dim])?;

        let bias_shard = if let Some(b) = full_bias {
            let b_vals = b.to_dtype(DType::F32)?.to_f32_vec()?;
            let bs = b_vals
                .get(row_start..row_start + shard_out)
                .ok_or(Error::IndexOutOfBounds {
                    index: row_start + shard_out,
                    len: b_vals.len(),
                })?;
            Some(Tensor::from_f32(bs, vec![1, shard_out])?)
        } else {
            None
        };

        Ok(Self {
            weight_shard,
            bias_shard,
            gather_output,
        })
    }

    /// Forward pass: `Y_shard = X @ W_shard^T [+ b_shard]`, optionally gathering along the last axis.
    pub fn forward<C: Communicator>(&self, x: &Tensor, comm: &C) -> Result<Tensor> {
        let wt = self.weight_shard.transpose(0, 1)?;
        let mut y = x.matmul(&wt)?;
        if let Some(b) = &self.bias_shard {
            y = y.add(b)?;
        }
        if !self.gather_output || comm.world_size() <= 1 {
            return Ok(y);
        }
        let shards = comm.all_gather(&y)?;
        concat_last_dim(&shards)
    }
}

/// Row-parallel linear projection sharded along the input dimension (`in_features / world_size`),
/// synchronizing partial sums with a single `AllReduce(Sum)`.
#[derive(Debug, Clone)]
pub struct RowParallelLinear {
    /// Local weight shard `[out_features, in_shard]`.
    pub weight_shard: Tensor,
    /// Optional full bias `[1, out_features]` applied after `AllReduce(Sum)`.
    pub bias: Option<Tensor>,
}

impl RowParallelLinear {
    /// Shards a full `[out_features, in_features]` weight along `in_features` for `rank / world_size`.
    pub fn from_full_weights(
        full_weight: &Tensor,
        full_bias: Option<&Tensor>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(Error::InvalidArgument {
                context: "RowParallelLinear::from_full_weights",
                reason: format!("invalid rank {rank} for world_size {world_size}"),
            });
        }
        let out_dim = full_weight.shape().dim(0)?;
        let in_dim = full_weight.shape().dim(1)?;
        if in_dim % world_size != 0 {
            return Err(Error::InvalidArgument {
                context: "RowParallelLinear::from_full_weights",
                reason: format!("in_features ({in_dim}) must be divisible by world_size ({world_size})"),
            });
        }
        let shard_in = in_dim / world_size;
        let col_start = rank * shard_in;
        let full_vals = full_weight.to_dtype(DType::F32)?.to_f32_vec()?;
        let mut shard_vals = Vec::with_capacity(out_dim * shard_in);
        for r in 0..out_dim {
            let row_off = r * in_dim + col_start;
            let slice = full_vals
                .get(row_off..row_off + shard_in)
                .ok_or(Error::IndexOutOfBounds {
                    index: row_off + shard_in,
                    len: full_vals.len(),
                })?;
            shard_vals.extend_from_slice(slice);
        }
        let weight_shard = Tensor::from_f32(&shard_vals, vec![out_dim, shard_in])?;
        let bias = full_bias.cloned();
        Ok(Self { weight_shard, bias })
    }

    /// Forward pass: `Y = AllReduceSum(X_shard @ W_shard^T) [+ bias]`.
    pub fn forward<C: Communicator>(&self, x_shard: &Tensor, comm: &C) -> Result<Tensor> {
        let wt = self.weight_shard.transpose(0, 1)?;
        let partial = x_shard.matmul(&wt)?;
        let mut out = comm.all_reduce(&partial, AllReduceOp::Sum)?;
        if let Some(b) = &self.bias {
            out = out.add(b)?;
        }
        Ok(out)
    }
}

/// Complete Megatron-LM style 2-layer Tensor Parallel MLP (`ColumnParallel -> ReLU -> RowParallel`),
/// requiring only a single `AllReduce(Sum)` per forward pass.
#[derive(Debug, Clone)]
pub struct TensorParallelMlp {
    /// First projection sharded by columns.
    pub fc1: ColumnParallelLinear,
    /// Second projection sharded by rows.
    pub fc2: RowParallelLinear,
}

impl TensorParallelMlp {
    /// Builds a [`TensorParallelMlp`] shard for `rank / world_size` from full weights.
    pub fn from_full_weights(
        w1: &Tensor,
        b1: Option<&Tensor>,
        w2: &Tensor,
        b2: Option<&Tensor>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let fc1 = ColumnParallelLinear::from_full_weights(w1, b1, rank, world_size, false)?;
        let fc2 = RowParallelLinear::from_full_weights(w2, b2, rank, world_size)?;
        Ok(Self { fc1, fc2 })
    }

    /// Executes `RowParallel(ReLU(ColumnParallel(x)))` across `comm`.
    pub fn forward<C: Communicator>(&self, x: &Tensor, comm: &C) -> Result<Tensor> {
        let h = self.fc1.forward(x, comm)?.relu()?;
        self.fc2.forward(&h, comm)
    }
}

fn concat_last_dim(shards: &[Tensor]) -> Result<Tensor> {
    let first = shards.first().ok_or(Error::InvalidArgument {
        context: "concat_last_dim",
        reason: "empty shard list".to_owned(),
    })?;
    if first.rank() != 2 {
        return Err(Error::InvalidRank {
            context: "concat_last_dim",
            rank: first.rank(),
            expected: 2,
        });
    }
    let rows = first.shape().dim(0)?;
    let shard_cols = first.shape().dim(1)?;
    let total_cols = shard_cols * shards.len();

    let shard_vecs: Vec<Vec<f32>> = shards
        .iter()
        .map(|s| s.to_dtype(DType::F32)?.to_f32_vec())
        .collect::<Result<Vec<_>>>()?;

    let mut out = Vec::with_capacity(rows * total_cols);
    for r in 0..rows {
        for sv in &shard_vecs {
            let start = r * shard_cols;
            let slice = sv
                .get(start..start + shard_cols)
                .ok_or(Error::IndexOutOfBounds {
                    index: start + shard_cols,
                    len: sv.len(),
                })?;
            out.extend_from_slice(slice);
        }
    }
    Tensor::from_f32(&out, vec![rows, total_cols])
}
