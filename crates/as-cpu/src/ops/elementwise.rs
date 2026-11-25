//! Element-wise binary and unary CPU kernels with fast contiguous paths
//! and general strided/broadcasted paths.

use as_backend::{BinaryOp, Storage, UnaryOp};
use as_core::{BF16, DType, Error, F16, Layout, MemoryLimits, Result};

use crate::storage::{CpuBuffer, CpuStorage};

/// Executes a binary element-wise operation on two CPU views of identical shape.
pub fn binary_op(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_layout: &Layout,
    rhs: &CpuStorage,
    rhs_layout: &Layout,
    limits: MemoryLimits,
) -> Result<CpuStorage> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::DTypeMismatch {
            expected: lhs.dtype(),
            actual: rhs.dtype(),
        });
    }
    if lhs_layout.shape() != rhs_layout.shape() {
        return Err(Error::ShapeMismatch {
            context: "cpu::binary_op",
            expected: lhs_layout.shape().dims().to_vec(),
            actual: rhs_layout.shape().dims().to_vec(),
        });
    }

    let numel = lhs_layout.numel();
    limits.check_allocation(numel, lhs.dtype())?;

    let both_contig = lhs_layout.is_contiguous() && rhs_layout.is_contiguous();

    let buf = match (lhs.buffer(), rhs.buffer()) {
        (CpuBuffer::F32(a), CpuBuffer::F32(b)) => {
            CpuBuffer::F32(binary_float_f32(op, a, lhs_layout, b, rhs_layout, both_contig)?)
        }
        (CpuBuffer::F64(a), CpuBuffer::F64(b)) => {
            CpuBuffer::F64(binary_float_f64(op, a, lhs_layout, b, rhs_layout, both_contig)?)
        }
        (CpuBuffer::F16(a), CpuBuffer::F16(b)) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, lhs_layout.physical_index(i)?)?.to_f32();
                let y = get_elem(b, rhs_layout.physical_index(i)?)?.to_f32();
                out.push(F16::from_f32(apply_bin_f32(op, x, y)));
            }
            CpuBuffer::F16(out)
        }
        (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, lhs_layout.physical_index(i)?)?.to_f32();
                let y = get_elem(b, rhs_layout.physical_index(i)?)?.to_f32();
                out.push(BF16::from_f32(apply_bin_f32(op, x, y)));
            }
            CpuBuffer::BF16(out)
        }
        (CpuBuffer::I32(a), CpuBuffer::I32(b)) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, lhs_layout.physical_index(i)?)?;
                let y = get_elem(b, rhs_layout.physical_index(i)?)?;
                out.push(apply_bin_i32(op, x, y)?);
            }
            CpuBuffer::I32(out)
        }
        (CpuBuffer::I64(a), CpuBuffer::I64(b)) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, lhs_layout.physical_index(i)?)?;
                let y = get_elem(b, rhs_layout.physical_index(i)?)?;
                out.push(apply_bin_i64(op, x, y)?);
            }
            CpuBuffer::I64(out)
        }
        _ => {
            return Err(Error::InternalInvariant {
                message: "mismatched CpuBuffer variants after dtype check",
            });
        }
    };

    Ok(CpuStorage::from_buffer(buf))
}

/// Executes a unary element-wise operation on a CPU view.
pub fn unary_op(
    op: UnaryOp,
    src: &CpuStorage,
    layout: &Layout,
    limits: MemoryLimits,
) -> Result<CpuStorage> {
    let numel = layout.numel();
    limits.check_allocation(numel, src.dtype())?;

    let buf = match src.buffer() {
        CpuBuffer::F32(a) => {
            let mut out = Vec::with_capacity(numel);
            if layout.is_contiguous() {
                let slice = contiguous_slice(a, layout)?;
                for &x in slice {
                    out.push(apply_unary_f32(op, x));
                }
            } else {
                for i in 0..numel {
                    let x = get_elem(a, layout.physical_index(i)?)?;
                    out.push(apply_unary_f32(op, x));
                }
            }
            CpuBuffer::F32(out)
        }
        CpuBuffer::F64(a) => {
            let mut out = Vec::with_capacity(numel);
            if layout.is_contiguous() {
                let slice = contiguous_slice(a, layout)?;
                for &x in slice {
                    out.push(apply_unary_f64(op, x));
                }
            } else {
                for i in 0..numel {
                    let x = get_elem(a, layout.physical_index(i)?)?;
                    out.push(apply_unary_f64(op, x));
                }
            }
            CpuBuffer::F64(out)
        }
        CpuBuffer::F16(a) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, layout.physical_index(i)?)?.to_f32();
                out.push(F16::from_f32(apply_unary_f32(op, x)));
            }
            CpuBuffer::F16(out)
        }
        CpuBuffer::BF16(a) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, layout.physical_index(i)?)?.to_f32();
                out.push(BF16::from_f32(apply_unary_f32(op, x)));
            }
            CpuBuffer::BF16(out)
        }
        CpuBuffer::I32(a) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, layout.physical_index(i)?)?;
                let res = match op {
                    UnaryOp::Neg => x.checked_neg().ok_or(Error::ArithmeticOverflow {
                        context: "i32::neg",
                    })?,
                    UnaryOp::Abs => x.checked_abs().ok_or(Error::ArithmeticOverflow {
                        context: "i32::abs",
                    })?,
                    UnaryOp::Relu => x.max(0),
                    UnaryOp::Exp | UnaryOp::Ln | UnaryOp::Sqrt | UnaryOp::Powf(_) => {
                        return Err(Error::UnsupportedDType {
                            op: unary_op_name(op),
                            dtype: DType::I32,
                        });
                    }
                };
                out.push(res);
            }
            CpuBuffer::I32(out)
        }
        CpuBuffer::I64(a) => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let x = get_elem(a, layout.physical_index(i)?)?;
                let res = match op {
                    UnaryOp::Neg => x.checked_neg().ok_or(Error::ArithmeticOverflow {
                        context: "i64::neg",
                    })?,
                    UnaryOp::Abs => x.checked_abs().ok_or(Error::ArithmeticOverflow {
                        context: "i64::abs",
                    })?,
                    UnaryOp::Relu => x.max(0),
                    UnaryOp::Exp | UnaryOp::Ln | UnaryOp::Sqrt | UnaryOp::Powf(_) => {
                        return Err(Error::UnsupportedDType {
                            op: unary_op_name(op),
                            dtype: DType::I64,
                        });
                    }
                };
                out.push(res);
            }
            CpuBuffer::I64(out)
        }
    };

    Ok(CpuStorage::from_buffer(buf))
}

fn binary_float_f32(
    op: BinaryOp,
    a: &[f32],
    a_layout: &Layout,
    b: &[f32],
    b_layout: &Layout,
    both_contig: bool,
) -> Result<Vec<f32>> {
    let numel = a_layout.numel();
    let mut out = Vec::with_capacity(numel);
    if both_contig {
        let sa = contiguous_slice(a, a_layout)?;
        let sb = contiguous_slice(b, b_layout)?;
        match op {
            BinaryOp::Add => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x + y)),
            BinaryOp::Sub => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x - y)),
            BinaryOp::Mul => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x * y)),
            BinaryOp::Div => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x / y)),
        }
    } else if is_last_dim_row_broadcast(a_layout, b_layout) {
        // Fast vectorized row-broadcast path (e.g. [M, N] + [1, N] bias addition)
        let row_len = a_layout.shape().dims().last().copied().unwrap_or(1);
        let sa = contiguous_slice(a, a_layout)?;
        let b_start = b_layout.offset();
        let sb = b
            .get(b_start..b_start + row_len)
            .ok_or(Error::IndexOutOfBounds {
                index: b_start + row_len,
                len: b.len(),
            })?;
        for row in sa.chunks_exact(row_len) {
            match op {
                BinaryOp::Add => out.extend(row.iter().zip(sb.iter()).map(|(&x, &y)| x + y)),
                BinaryOp::Sub => out.extend(row.iter().zip(sb.iter()).map(|(&x, &y)| x - y)),
                BinaryOp::Mul => out.extend(row.iter().zip(sb.iter()).map(|(&x, &y)| x * y)),
                BinaryOp::Div => out.extend(row.iter().zip(sb.iter()).map(|(&x, &y)| x / y)),
            }
        }
    } else {
        for i in 0..numel {
            let x = get_elem(a, a_layout.physical_index(i)?)?;
            let y = get_elem(b, b_layout.physical_index(i)?)?;
            out.push(apply_bin_f32(op, x, y));
        }
    }
    Ok(out)
}

fn binary_float_f64(
    op: BinaryOp,
    a: &[f64],
    a_layout: &Layout,
    b: &[f64],
    b_layout: &Layout,
    both_contig: bool,
) -> Result<Vec<f64>> {
    let numel = a_layout.numel();
    let mut out = Vec::with_capacity(numel);
    if both_contig {
        let sa = contiguous_slice(a, a_layout)?;
        let sb = contiguous_slice(b, b_layout)?;
        match op {
            BinaryOp::Add => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x + y)),
            BinaryOp::Sub => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x - y)),
            BinaryOp::Mul => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x * y)),
            BinaryOp::Div => out.extend(sa.iter().zip(sb.iter()).map(|(&x, &y)| x / y)),
        }
    } else {
        for i in 0..numel {
            let x = get_elem(a, a_layout.physical_index(i)?)?;
            let y = get_elem(b, b_layout.physical_index(i)?)?;
            out.push(apply_bin_f64(op, x, y));
        }
    }
    Ok(out)
}

#[inline]
fn apply_bin_f32(op: BinaryOp, x: f32, y: f32) -> f32 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
    }
}

#[inline]
fn apply_bin_f64(op: BinaryOp, x: f64, y: f64) -> f64 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
    }
}

#[inline]
fn apply_bin_i32(op: BinaryOp, x: i32, y: i32) -> Result<i32> {
    match op {
        BinaryOp::Add => x.checked_add(y).ok_or(Error::ArithmeticOverflow {
            context: "i32::add",
        }),
        BinaryOp::Sub => x.checked_sub(y).ok_or(Error::ArithmeticOverflow {
            context: "i32::sub",
        }),
        BinaryOp::Mul => x.checked_mul(y).ok_or(Error::ArithmeticOverflow {
            context: "i32::mul",
        }),
        BinaryOp::Div => {
            if y == 0 {
                return Err(Error::InvalidArgument {
                    context: "i32::div",
                    reason: "integer division by zero".to_owned(),
                });
            }
            x.checked_div(y).ok_or(Error::ArithmeticOverflow {
                context: "i32::div",
            })
        }
    }
}

#[inline]
fn apply_bin_i64(op: BinaryOp, x: i64, y: i64) -> Result<i64> {
    match op {
        BinaryOp::Add => x.checked_add(y).ok_or(Error::ArithmeticOverflow {
            context: "i64::add",
        }),
        BinaryOp::Sub => x.checked_sub(y).ok_or(Error::ArithmeticOverflow {
            context: "i64::sub",
        }),
        BinaryOp::Mul => x.checked_mul(y).ok_or(Error::ArithmeticOverflow {
            context: "i64::mul",
        }),
        BinaryOp::Div => {
            if y == 0 {
                return Err(Error::InvalidArgument {
                    context: "i64::div",
                    reason: "integer division by zero".to_owned(),
                });
            }
            x.checked_div(y).ok_or(Error::ArithmeticOverflow {
                context: "i64::div",
            })
        }
    }
}

#[inline]
fn apply_unary_f32(op: UnaryOp, x: f32) -> f32 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Relu => {
            if x.is_nan() {
                x
            } else {
                x.max(0.0)
            }
        }
        UnaryOp::Powf(p) => x.powf(p as f32),
    }
}

#[inline]
fn apply_unary_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Relu => {
            if x.is_nan() {
                x
            } else {
                x.max(0.0)
            }
        }
        UnaryOp::Powf(p) => x.powf(p),
    }
}

fn unary_op_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Abs => "abs",
        UnaryOp::Exp => "exp",
        UnaryOp::Ln => "ln",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Relu => "relu",
        UnaryOp::Powf(_) => "powf",
    }
}

#[inline]
fn get_elem<T: Copy>(data: &[T], index: usize) -> Result<T> {
    data.get(index).copied().ok_or(Error::IndexOutOfBounds {
        index,
        len: data.len(),
    })
}

fn contiguous_slice<'a, T>(data: &'a [T], layout: &Layout) -> Result<&'a [T]> {
    let start = layout.offset();
    let end = start
        .checked_add(layout.numel())
        .ok_or(Error::ArithmeticOverflow {
            context: "contiguous_slice",
        })?;
    data.get(start..end).ok_or(Error::IndexOutOfBounds {
        index: end,
        len: data.len(),
    })
}

/// Detects the hot pattern where `a_layout` is contiguous and `b_layout` broadcasts
/// a contiguous last-axis vector (all leading strides == 0, last stride == 1).
fn is_last_dim_row_broadcast(a_layout: &Layout, b_layout: &Layout) -> bool {
    let rank = a_layout.rank();
    if rank < 2 || !a_layout.is_contiguous() {
        return false;
    }
    let last_dim = a_layout.shape().dims().last().copied().unwrap_or(0);
    if last_dim == 0 {
        return false;
    }
    let b_strides = b_layout.strides();
    b_strides.last().copied() == Some(1)
        && b_strides
            .get(..rank - 1)
            .is_some_and(|lead| lead.iter().all(|&s| s == 0))
}
