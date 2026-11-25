//! Global and axis-wise CPU reduction kernels (`Sum`, `Mean`, `Max`, `Min`).

use as_backend::{ReduceOp, Storage};
use as_core::{BF16, DType, Error, F16, Layout, MemoryLimits, Result, Shape};

use crate::storage::{CpuBuffer, CpuStorage};

/// Executes a reduction globally (`axis == None`) or along a specific `axis`.
pub fn reduce(
    op: ReduceOp,
    src: &CpuStorage,
    layout: &Layout,
    axis: Option<usize>,
    keepdim: bool,
    limits: MemoryLimits,
) -> Result<(CpuStorage, Shape)> {
    let dtype = src.dtype();
    if op == ReduceOp::Mean && !dtype.is_float() {
        return Err(Error::UnsupportedDType { op: "mean", dtype });
    }

    let dims = layout.shape().dims();
    let rank = dims.len();

    let (out_shape, reduce_len) = match axis {
        None => {
            let out_dims = if keepdim { vec![1; rank] } else { Vec::new() };
            (Shape::new(out_dims)?, layout.numel())
        }
        Some(ax) => {
            if ax >= rank {
                return Err(Error::InvalidAxis { axis: ax, rank });
            }
            let rlen = dims.get(ax).copied().unwrap_or(0);
            let mut out_dims = dims.to_vec();
            if keepdim {
                if let Some(slot) = out_dims.get_mut(ax) {
                    *slot = 1;
                }
            } else {
                out_dims.remove(ax);
            }
            (Shape::new(out_dims)?, rlen)
        }
    };

    if reduce_len == 0 && matches!(op, ReduceOp::Max | ReduceOp::Min | ReduceOp::Mean) {
        return Err(Error::InvalidArgument {
            context: "reduce",
            reason: "cannot compute mean/max/min over a zero-length dimension".to_owned(),
        });
    }

    let out_numel = out_shape.numel();
    limits.check_allocation(out_numel, dtype)?;

    // Build index mapping: for each output element, collect its source physical indices.
    let buf = match src.buffer() {
        CpuBuffer::F32(data) => CpuBuffer::F32(reduce_f32(
            op, data, layout, axis, keepdim, out_numel, reduce_len,
        )?),
        CpuBuffer::F64(data) => CpuBuffer::F64(reduce_f64(
            op, data, layout, axis, keepdim, out_numel, reduce_len,
        )?),
        CpuBuffer::F16(data) => {
            // Accumulate in f32 for numerical accuracy
            let f32_data: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
            let res = reduce_f32(op, &f32_data, layout, axis, keepdim, out_numel, reduce_len)?;
            CpuBuffer::F16(res.into_iter().map(F16::from_f32).collect())
        }
        CpuBuffer::BF16(data) => {
            let f32_data: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
            let res = reduce_f32(op, &f32_data, layout, axis, keepdim, out_numel, reduce_len)?;
            CpuBuffer::BF16(res.into_iter().map(BF16::from_f32).collect())
        }
        CpuBuffer::I32(data) => CpuBuffer::I32(reduce_i32(
            op, data, layout, axis, keepdim, out_numel, reduce_len,
        )?),
        CpuBuffer::I64(data) => CpuBuffer::I64(reduce_i64(
            op, data, layout, axis, keepdim, out_numel, reduce_len,
        )?),
    };

    Ok((CpuStorage::from_buffer(buf), out_shape))
}

fn for_each_reduction_group<F>(
    layout: &Layout,
    axis: Option<usize>,
    keepdim: bool,
    out_numel: usize,
    reduce_len: usize,
    mut f: F,
) -> Result<()>
where
    F: FnMut(usize, usize) -> Result<()>,
{
    match axis {
        None => {
            for i in 0..reduce_len {
                let phys = layout.physical_index(i)?;
                f(0, phys)?;
            }
        }
        Some(ax) => {
            let dims = layout.shape().dims();
            let rank = dims.len();
            let mut out_dims = dims.to_vec();
            if keepdim {
                if let Some(slot) = out_dims.get_mut(ax) {
                    *slot = 1;
                }
            } else {
                out_dims.remove(ax);
            }
            let out_strides = Shape::new(out_dims)?.contiguous_strides();

            let mut coords = vec![0usize; rank];
            for out_idx in 0..out_numel {
                // Decode out_idx into coords (skipping ax)
                let mut rem = out_idx;
                let mut out_dim_pos = 0usize;
                for dim_idx in 0..rank {
                    if dim_idx == ax {
                        if keepdim {
                            out_dim_pos += 1;
                        }
                        continue;
                    }
                    let stride = out_strides.get(out_dim_pos).copied().unwrap_or(1);
                    let c = if stride == 0 { 0 } else { rem / stride };
                    rem = if stride == 0 { 0 } else { rem % stride };
                    if let Some(slot) = coords.get_mut(dim_idx) {
                        *slot = c;
                    }
                    out_dim_pos += 1;
                }

                for r in 0..reduce_len {
                    if let Some(slot) = coords.get_mut(ax) {
                        *slot = r;
                    }
                    let phys = layout.index_of(&coords)?;
                    f(out_idx, phys)?;
                }
            }
        }
    }
    Ok(())
}

fn reduce_f32(
    op: ReduceOp,
    data: &[f32],
    layout: &Layout,
    axis: Option<usize>,
    keepdim: bool,
    out_numel: usize,
    reduce_len: usize,
) -> Result<Vec<f32>> {
    // Accumulate sum/mean in f64 to reduce floating-point drift on large reductions.
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            let mut acc = vec![0.0f64; out_numel];
            for_each_reduction_group(layout, axis, keepdim, out_numel, reduce_len, |out_idx, phys| {
                let v = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: data.len(),
                })?;
                if let Some(slot) = acc.get_mut(out_idx) {
                    *slot += f64::from(v);
                }
                Ok(())
            })?;
            if op == ReduceOp::Mean {
                let denom = reduce_len as f64;
                Ok(acc.into_iter().map(|s| (s / denom) as f32).collect())
            } else {
                Ok(acc.into_iter().map(|s| s as f32).collect())
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            let init = if op == ReduceOp::Max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            let mut out = vec![init; out_numel];
            let mut initialized = vec![false; out_numel];
            for_each_reduction_group(layout, axis, keepdim, out_numel, reduce_len, |out_idx, phys| {
                let v = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: data.len(),
                })?;
                if let (Some(slot), Some(flag)) = (out.get_mut(out_idx), initialized.get_mut(out_idx)) {
                    if !*flag {
                        *slot = v;
                        *flag = true;
                    } else if v.is_nan() || slot.is_nan() {
                        *slot = f32::NAN;
                    } else if op == ReduceOp::Max {
                        if v > *slot {
                            *slot = v;
                        }
                    } else if v < *slot {
                        *slot = v;
                    }
                }
                Ok(())
            })?;
            Ok(out)
        }
    }
}

fn reduce_f64(
    op: ReduceOp,
    data: &[f64],
    layout: &Layout,
    axis: Option<usize>,
    keepdim: bool,
    out_numel: usize,
    reduce_len: usize,
) -> Result<Vec<f64>> {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            let mut acc = vec![0.0f64; out_numel];
            for_each_reduction_group(layout, axis, keepdim, out_numel, reduce_len, |out_idx, phys| {
                let v = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: data.len(),
                })?;
                if let Some(slot) = acc.get_mut(out_idx) {
                    *slot += v;
                }
                Ok(())
            })?;
            if op == ReduceOp::Mean {
                let denom = reduce_len as f64;
                for s in &mut acc {
                    *s /= denom;
                }
            }
            Ok(acc)
        }
        ReduceOp::Max | ReduceOp::Min => {
            let init = if op == ReduceOp::Max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            let mut out = vec![init; out_numel];
            let mut initialized = vec![false; out_numel];
            for_each_reduction_group(layout, axis, keepdim, out_numel, reduce_len, |out_idx, phys| {
                let v = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: data.len(),
                })?;
                if let (Some(slot), Some(flag)) = (out.get_mut(out_idx), initialized.get_mut(out_idx)) {
                    if !*flag {
                        *slot = v;
                        *flag = true;
                    } else if v.is_nan() || slot.is_nan() {
                        *slot = f64::NAN;
                    } else if op == ReduceOp::Max {
                        if v > *slot {
                            *slot = v;
                        }
                    } else if v < *slot {
                        *slot = v;
                    }
                }
                Ok(())
            })?;
            Ok(out)
        }
    }
}

fn reduce_i32(
    op: ReduceOp,
    data: &[i32],
    layout: &Layout,
    axis: Option<usize>,
    keepdim: bool,
    out_numel: usize,
    reduce_len: usize,
) -> Result<Vec<i32>> {
    let mut out = vec![0i32; out_numel];
    let mut initialized = vec![false; out_numel];
    for_each_reduction_group(layout, axis, keepdim, out_numel, reduce_len, |out_idx, phys| {
        let v = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
            index: phys,
            len: data.len(),
        })?;
        if let (Some(slot), Some(flag)) = (out.get_mut(out_idx), initialized.get_mut(out_idx)) {
            match op {
                ReduceOp::Sum => {
                    *slot = slot.checked_add(v).ok_or(Error::ArithmeticOverflow {
                        context: "reduce_i32 sum",
                    })?;
                }
                ReduceOp::Max => {
                    if !*flag || v > *slot {
                        *slot = v;
                        *flag = true;
                    }
                }
                ReduceOp::Min => {
                    if !*flag || v < *slot {
                        *slot = v;
                        *flag = true;
                    }
                }
                ReduceOp::Mean => {
                    return Err(Error::UnsupportedDType {
                        op: "mean",
                        dtype: DType::I32,
                    });
                }
            }
        }
        Ok(())
    })?;
    Ok(out)
}

fn reduce_i64(
    op: ReduceOp,
    data: &[i64],
    layout: &Layout,
    axis: Option<usize>,
    keepdim: bool,
    out_numel: usize,
    reduce_len: usize,
) -> Result<Vec<i64>> {
    let mut out = vec![0i64; out_numel];
    let mut initialized = vec![false; out_numel];
    for_each_reduction_group(layout, axis, keepdim, out_numel, reduce_len, |out_idx, phys| {
        let v = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
            index: phys,
            len: data.len(),
        })?;
        if let (Some(slot), Some(flag)) = (out.get_mut(out_idx), initialized.get_mut(out_idx)) {
            match op {
                ReduceOp::Sum => {
                    *slot = slot.checked_add(v).ok_or(Error::ArithmeticOverflow {
                        context: "reduce_i64 sum",
                    })?;
                }
                ReduceOp::Max => {
                    if !*flag || v > *slot {
                        *slot = v;
                        *flag = true;
                    }
                }
                ReduceOp::Min => {
                    if !*flag || v < *slot {
                        *slot = v;
                        *flag = true;
                    }
                }
                ReduceOp::Mean => {
                    return Err(Error::UnsupportedDType {
                        op: "mean",
                        dtype: DType::I64,
                    });
                }
            }
        }
        Ok(())
    })?;
    Ok(out)
}
