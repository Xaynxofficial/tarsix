//! Cache-blocked (tiled) CPU matrix multiplication for 2D and batched N-D tensors.
//!
//! Uses `i-k-j` tiled loop ordering (`TILE = 64`) for sequential row access and
//! strong LLVM auto-vectorization without any `unsafe` code.
//! `F16` and `BF16` accumulate in `f32` for numerical stability.

use as_backend::Storage;
use as_core::{BF16, Error, F16, Layout, MemoryLimits, Result, Shape};

use crate::storage::{CpuBuffer, CpuStorage};

const TILE: usize = 64;

/// Executes 2D or batched N-D matrix multiplication with broadcastable batch dimensions.
pub fn matmul(
    lhs: &CpuStorage,
    lhs_layout: &Layout,
    rhs: &CpuStorage,
    rhs_layout: &Layout,
    limits: MemoryLimits,
) -> Result<(CpuStorage, Shape)> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::DTypeMismatch {
            expected: lhs.dtype(),
            actual: rhs.dtype(),
        });
    }

    let lhs_rank = lhs_layout.rank();
    let rhs_rank = rhs_layout.rank();
    if lhs_rank < 2 {
        return Err(Error::InvalidRank {
            context: "matmul lhs",
            rank: lhs_rank,
            expected: 2,
        });
    }
    if rhs_rank < 2 {
        return Err(Error::InvalidRank {
            context: "matmul rhs",
            rank: rhs_rank,
            expected: 2,
        });
    }

    let lhs_dims = lhs_layout.shape().dims();
    let rhs_dims = rhs_layout.shape().dims();
    let m = lhs_dims.get(lhs_rank - 2).copied().unwrap_or(0);
    let k_lhs = lhs_dims.get(lhs_rank - 1).copied().unwrap_or(0);
    let k_rhs = rhs_dims.get(rhs_rank - 2).copied().unwrap_or(0);
    let n = rhs_dims.get(rhs_rank - 1).copied().unwrap_or(0);

    if k_lhs != k_rhs {
        return Err(Error::ShapeMismatch {
            context: "matmul inner dimension K",
            expected: lhs_dims.to_vec(),
            actual: rhs_dims.to_vec(),
        });
    }
    let k = k_lhs;

    // Broadcast batch dimensions
    let lhs_batch = Shape::new(lhs_dims.get(..lhs_rank - 2).unwrap_or(&[]).to_vec())?;
    let rhs_batch = Shape::new(rhs_dims.get(..rhs_rank - 2).unwrap_or(&[]).to_vec())?;
    let out_batch = lhs_batch.broadcast_shape(&rhs_batch)?;

    let mut out_dims = out_batch.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::new(out_dims)?;

    let out_numel = out_shape.numel();
    limits.check_allocation(out_numel, lhs.dtype())?;

    // Broadcast lhs and rhs batch dimensions to match out_batch
    let mut lhs_bcast_dims = out_batch.dims().to_vec();
    lhs_bcast_dims.push(m);
    lhs_bcast_dims.push(k);
    let lhs_bcast = lhs_layout.broadcast_to(&Shape::new(lhs_bcast_dims)?)?;

    let mut rhs_bcast_dims = out_batch.dims().to_vec();
    rhs_bcast_dims.push(k);
    rhs_bcast_dims.push(n);
    let rhs_bcast = rhs_layout.broadcast_to(&Shape::new(rhs_bcast_dims)?)?;

    // Materialize contiguous batch-packed buffers so the tiled GEMM kernel runs on linear slices
    let lhs_contig = lhs.contiguous(&lhs_bcast, limits)?;
    let rhs_contig = rhs.contiguous(&rhs_bcast, limits)?;

    let batch_count = out_batch.numel();
    let lhs_mat_stride = m.saturating_mul(k);
    let rhs_mat_stride = k.saturating_mul(n);
    let out_mat_stride = m.saturating_mul(n);

    let out_buf = match (lhs_contig.buffer(), rhs_contig.buffer()) {
        (CpuBuffer::F32(a), CpuBuffer::F32(b)) => {
            let mut c = vec![0.0f32; out_numel];
            for batch in 0..batch_count {
                let a_slice = slice_matrix(a, batch, lhs_mat_stride)?;
                let b_slice = slice_matrix(b, batch, rhs_mat_stride)?;
                let c_slice = slice_matrix_mut(&mut c, batch, out_mat_stride)?;
                gemm_f32_tiled(a_slice, b_slice, c_slice, m, k, n);
            }
            CpuBuffer::F32(c)
        }
        (CpuBuffer::F64(a), CpuBuffer::F64(b)) => {
            let mut c = vec![0.0f64; out_numel];
            for batch in 0..batch_count {
                let a_slice = slice_matrix(a, batch, lhs_mat_stride)?;
                let b_slice = slice_matrix(b, batch, rhs_mat_stride)?;
                let c_slice = slice_matrix_mut(&mut c, batch, out_mat_stride)?;
                gemm_f64_tiled(a_slice, b_slice, c_slice, m, k, n);
            }
            CpuBuffer::F64(c)
        }
        (CpuBuffer::F16(a), CpuBuffer::F16(b)) => {
            // Upcast to f32 for numerically stable accumulation, then round back to f16.
            let a_f32: Vec<f32> = a.iter().map(|&x| x.to_f32()).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| x.to_f32()).collect();
            let mut c_f32 = vec![0.0f32; out_numel];
            for batch in 0..batch_count {
                let a_slice = slice_matrix(&a_f32, batch, lhs_mat_stride)?;
                let b_slice = slice_matrix(&b_f32, batch, rhs_mat_stride)?;
                let c_slice = slice_matrix_mut(&mut c_f32, batch, out_mat_stride)?;
                gemm_f32_tiled(a_slice, b_slice, c_slice, m, k, n);
            }
            CpuBuffer::F16(c_f32.into_iter().map(F16::from_f32).collect())
        }
        (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => {
            let a_f32: Vec<f32> = a.iter().map(|&x| x.to_f32()).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| x.to_f32()).collect();
            let mut c_f32 = vec![0.0f32; out_numel];
            for batch in 0..batch_count {
                let a_slice = slice_matrix(&a_f32, batch, lhs_mat_stride)?;
                let b_slice = slice_matrix(&b_f32, batch, rhs_mat_stride)?;
                let c_slice = slice_matrix_mut(&mut c_f32, batch, out_mat_stride)?;
                gemm_f32_tiled(a_slice, b_slice, c_slice, m, k, n);
            }
            CpuBuffer::BF16(c_f32.into_iter().map(BF16::from_f32).collect())
        }
        (CpuBuffer::I32(a), CpuBuffer::I32(b)) => {
            let mut c = vec![0i32; out_numel];
            for batch in 0..batch_count {
                let a_slice = slice_matrix(a, batch, lhs_mat_stride)?;
                let b_slice = slice_matrix(b, batch, rhs_mat_stride)?;
                let c_slice = slice_matrix_mut(&mut c, batch, out_mat_stride)?;
                gemm_i32_checked(a_slice, b_slice, c_slice, m, k, n)?;
            }
            CpuBuffer::I32(c)
        }
        (CpuBuffer::I64(a), CpuBuffer::I64(b)) => {
            let mut c = vec![0i64; out_numel];
            for batch in 0..batch_count {
                let a_slice = slice_matrix(a, batch, lhs_mat_stride)?;
                let b_slice = slice_matrix(b, batch, rhs_mat_stride)?;
                let c_slice = slice_matrix_mut(&mut c, batch, out_mat_stride)?;
                gemm_i64_checked(a_slice, b_slice, c_slice, m, k, n)?;
            }
            CpuBuffer::I64(c)
        }
        _ => {
            return Err(Error::InternalInvariant {
                message: "mismatched CpuBuffer variants in matmul",
            });
        }
    };

    Ok((CpuStorage::from_buffer(out_buf), out_shape))
}

/// Cache-blocked `i-k-j` matrix multiplication for `f32` with safe multi-core
/// row-band parallelism via `std::thread::scope` (zero `unsafe`).
fn gemm_f32_tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    // Parallelize across disjoint row bands when workload justifies thread dispatch
    let work = m.saturating_mul(k).saturating_mul(n);
    if threads > 1 && m >= 64 && work >= 1_000_000 {
        let num_workers = threads.min(m / 16).max(1);
        let rows_per_worker = m.div_ceil(num_workers);
        std::thread::scope(|s| {
            for (worker_idx, c_band) in c.chunks_mut(rows_per_worker * n).enumerate() {
                let row_start = worker_idx * rows_per_worker;
                let band_rows = c_band.len() / n;
                let a_band = a
                    .get(row_start * k..(row_start + band_rows) * k)
                    .unwrap_or(&[]);
                s.spawn(move || {
                    gemm_f32_tiled_band(a_band, b, c_band, band_rows, k, n);
                });
            }
        });
    } else {
        gemm_f32_tiled_band(a, b, c, m, k, n);
    }
}

fn gemm_f32_tiled_band(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i0 in (0..m).step_by(TILE) {
        let i_max = (i0 + TILE).min(m);
        for k0 in (0..k).step_by(TILE) {
            let k_max = (k0 + TILE).min(k);
            for j0 in (0..n).step_by(TILE) {
                let j_max = (j0 + TILE).min(n);
                for i in i0..i_max {
                    let c_row_start = i * n + j0;
                    let c_row_end = i * n + j_max;
                    let Some(c_tile) = c.get_mut(c_row_start..c_row_end) else {
                        continue;
                    };
                    for p in k0..k_max {
                        let Some(&a_val) = a.get(i * k + p) else {
                            continue;
                        };
                        let b_row_start = p * n + j0;
                        let b_row_end = p * n + j_max;
                        let Some(b_tile) = b.get(b_row_start..b_row_end) else {
                            continue;
                        };
                        for (c_cell, &b_val) in c_tile.iter_mut().zip(b_tile.iter()) {
                            *c_cell += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
}

/// Cache-blocked `i-k-j` matrix multiplication for `f64`.
fn gemm_f64_tiled(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    for i0 in (0..m).step_by(TILE) {
        let i_max = (i0 + TILE).min(m);
        for k0 in (0..k).step_by(TILE) {
            let k_max = (k0 + TILE).min(k);
            for j0 in (0..n).step_by(TILE) {
                let j_max = (j0 + TILE).min(n);
                for i in i0..i_max {
                    let c_row_start = i * n + j0;
                    let c_row_end = i * n + j_max;
                    let Some(c_tile) = c.get_mut(c_row_start..c_row_end) else {
                        continue;
                    };
                    for p in k0..k_max {
                        let Some(&a_val) = a.get(i * k + p) else {
                            continue;
                        };
                        let b_row_start = p * n + j0;
                        let b_row_end = p * n + j_max;
                        let Some(b_tile) = b.get(b_row_start..b_row_end) else {
                            continue;
                        };
                        for (c_cell, &b_val) in c_tile.iter_mut().zip(b_tile.iter()) {
                            *c_cell += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
}

fn gemm_i32_checked(
    a: &[i32],
    b: &[i32],
    c: &mut [i32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    for i in 0..m {
        for p in 0..k {
            let a_val = a.get(i * k + p).copied().ok_or(Error::IndexOutOfBounds {
                index: i * k + p,
                len: a.len(),
            })?;
            for j in 0..n {
                let b_val = b.get(p * n + j).copied().ok_or(Error::IndexOutOfBounds {
                    index: p * n + j,
                    len: b.len(),
                })?;
                let prod = a_val.checked_mul(b_val).ok_or(Error::ArithmeticOverflow {
                    context: "matmul i32 multiply",
                })?;
                let c_len = c.len();
                let c_cell = c.get_mut(i * n + j).ok_or(Error::IndexOutOfBounds {
                    index: i * n + j,
                    len: c_len,
                })?;
                *c_cell = c_cell.checked_add(prod).ok_or(Error::ArithmeticOverflow {
                    context: "matmul i32 accumulate",
                })?;
            }
        }
    }
    Ok(())
}

fn gemm_i64_checked(
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    for i in 0..m {
        for p in 0..k {
            let a_val = a.get(i * k + p).copied().ok_or(Error::IndexOutOfBounds {
                index: i * k + p,
                len: a.len(),
            })?;
            for j in 0..n {
                let b_val = b.get(p * n + j).copied().ok_or(Error::IndexOutOfBounds {
                    index: p * n + j,
                    len: b.len(),
                })?;
                let prod = a_val.checked_mul(b_val).ok_or(Error::ArithmeticOverflow {
                    context: "matmul i64 multiply",
                })?;
                let c_len = c.len();
                let c_cell = c.get_mut(i * n + j).ok_or(Error::IndexOutOfBounds {
                    index: i * n + j,
                    len: c_len,
                })?;
                *c_cell = c_cell.checked_add(prod).ok_or(Error::ArithmeticOverflow {
                    context: "matmul i64 accumulate",
                })?;
            }
        }
    }
    Ok(())
}

fn slice_matrix<T>(data: &[T], batch: usize, mat_stride: usize) -> Result<&[T]> {
    let start = batch
        .checked_mul(mat_stride)
        .ok_or(Error::ArithmeticOverflow {
            context: "matmul batch offset",
        })?;
    let end = start
        .checked_add(mat_stride)
        .ok_or(Error::ArithmeticOverflow {
            context: "matmul batch end",
        })?;
    data.get(start..end).ok_or(Error::IndexOutOfBounds {
        index: end,
        len: data.len(),
    })
}

fn slice_matrix_mut<T>(data: &mut [T], batch: usize, mat_stride: usize) -> Result<&mut [T]> {
    let start = batch
        .checked_mul(mat_stride)
        .ok_or(Error::ArithmeticOverflow {
            context: "matmul batch offset",
        })?;
    let end = start
        .checked_add(mat_stride)
        .ok_or(Error::ArithmeticOverflow {
            context: "matmul batch end",
        })?;
    let len = data.len();
    data.get_mut(start..end)
        .ok_or(Error::IndexOutOfBounds { index: end, len })
}
