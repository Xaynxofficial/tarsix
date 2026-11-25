//! Reference-grade, memory-safe CPU storage implementation.
//!
//! Backed by `Arc<CpuBuffer>` so that zero-copy view operations (`transpose`,
//! `reshape`, `slice`, `broadcast_to`) share underlying storage without cloning memory.

use std::sync::Arc;

use as_backend::Storage;
use as_core::{BF16, DType, Device, Error, F16, Layout, MemoryLimits, Result};

/// Typed contiguous CPU memory buffer.
#[derive(Debug, Clone)]
pub enum CpuBuffer {
    /// 32-bit float buffer.
    F32(Vec<f32>),
    /// 64-bit float buffer.
    F64(Vec<f64>),
    /// 16-bit IEEE-754 float buffer.
    F16(Vec<F16>),
    /// 16-bit bfloat16 buffer.
    BF16(Vec<BF16>),
    /// 32-bit signed integer buffer.
    I32(Vec<i32>),
    /// 64-bit signed integer buffer.
    I64(Vec<i64>),
}

/// Shared CPU storage handle.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    buffer: Arc<CpuBuffer>,
}

impl CpuStorage {
    /// Wraps a [`CpuBuffer`] in a shared [`CpuStorage`].
    pub fn from_buffer(buffer: CpuBuffer) -> Self {
        Self {
            buffer: Arc::new(buffer),
        }
    }

    /// Borrows the underlying [`CpuBuffer`].
    pub fn buffer(&self) -> &CpuBuffer {
        &self.buffer
    }

    /// Allocates a buffer filled with `value` converted to `dtype`, enforcing `limits`.
    pub fn full_f64(numel: usize, value: f64, dtype: DType, limits: MemoryLimits) -> Result<Self> {
        limits.check_allocation(numel, dtype)?;
        let buffer = match dtype {
            DType::F32 => CpuBuffer::F32(vec![value as f32; numel]),
            DType::F64 => CpuBuffer::F64(vec![value; numel]),
            DType::F16 => CpuBuffer::F16(vec![F16::from_f32(value as f32); numel]),
            DType::BF16 => CpuBuffer::BF16(vec![BF16::from_f32(value as f32); numel]),
            DType::I32 => CpuBuffer::I32(vec![value as i32; numel]),
            DType::I64 => CpuBuffer::I64(vec![value as i64; numel]),
        };
        Ok(Self::from_buffer(buffer))
    }

    /// Materializes any layout into a contiguous `Vec<T>` via validated indexing.
    pub(crate) fn materialize_slice<T: Copy>(
        data: &[T],
        layout: &Layout,
        limits: MemoryLimits,
        dtype: DType,
    ) -> Result<Vec<T>> {
        let numel = layout.numel();
        limits.check_allocation(numel, dtype)?;
        if layout.is_contiguous() {
            let start = layout.offset();
            let end = start.checked_add(numel).ok_or(Error::ArithmeticOverflow {
                context: "CpuStorage::materialize_slice bounds",
            })?;
            let slice = data.get(start..end).ok_or(Error::IndexOutOfBounds {
                index: end,
                len: data.len(),
            })?;
            return Ok(slice.to_vec());
        }

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let phys = layout.physical_index(i)?;
            let val = data.get(phys).copied().ok_or(Error::IndexOutOfBounds {
                index: phys,
                len: data.len(),
            })?;
            out.push(val);
        }
        Ok(out)
    }

    /// Returns a contiguous `CpuStorage` for `layout`.
    pub fn contiguous(&self, layout: &Layout, limits: MemoryLimits) -> Result<Self> {
        if layout.is_contiguous() && layout.offset() == 0 && layout.numel() == self.len() {
            return Ok(self.clone());
        }
        let buf = match self.buffer() {
            CpuBuffer::F32(v) => CpuBuffer::F32(Self::materialize_slice(v, layout, limits, DType::F32)?),
            CpuBuffer::F64(v) => CpuBuffer::F64(Self::materialize_slice(v, layout, limits, DType::F64)?),
            CpuBuffer::F16(v) => CpuBuffer::F16(Self::materialize_slice(v, layout, limits, DType::F16)?),
            CpuBuffer::BF16(v) => CpuBuffer::BF16(Self::materialize_slice(v, layout, limits, DType::BF16)?),
            CpuBuffer::I32(v) => CpuBuffer::I32(Self::materialize_slice(v, layout, limits, DType::I32)?),
            CpuBuffer::I64(v) => CpuBuffer::I64(Self::materialize_slice(v, layout, limits, DType::I64)?),
        };
        Ok(Self::from_buffer(buf))
    }

    /// Casts elements to `target_dtype` in logical layout order.
    pub fn cast(&self, layout: &Layout, target_dtype: DType, limits: MemoryLimits) -> Result<Self> {
        if self.dtype() == target_dtype {
            return self.contiguous(layout, limits);
        }
        let numel = layout.numel();
        limits.check_allocation(numel, target_dtype)?;

        // Convert source elements to f64 (or preserve i64 when casting i64 -> i64 handled above)
        let mut as_f64 = Vec::with_capacity(numel);
        for i in 0..numel {
            let phys = layout.physical_index(i)?;
            let val = match self.buffer() {
                CpuBuffer::F32(v) => f64::from(*v.get(phys).ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: v.len(),
                })?),
                CpuBuffer::F64(v) => *v.get(phys).ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: v.len(),
                })?,
                CpuBuffer::F16(v) => f64::from(
                    v.get(phys)
                        .ok_or(Error::IndexOutOfBounds {
                            index: phys,
                            len: v.len(),
                        })?
                        .to_f32(),
                ),
                CpuBuffer::BF16(v) => f64::from(
                    v.get(phys)
                        .ok_or(Error::IndexOutOfBounds {
                            index: phys,
                            len: v.len(),
                        })?
                        .to_f32(),
                ),
                CpuBuffer::I32(v) => f64::from(*v.get(phys).ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: v.len(),
                })?),
                CpuBuffer::I64(v) => *v.get(phys).ok_or(Error::IndexOutOfBounds {
                    index: phys,
                    len: v.len(),
                })? as f64,
            };
            as_f64.push(val);
        }

        let buf = match target_dtype {
            DType::F32 => CpuBuffer::F32(as_f64.into_iter().map(|x| x as f32).collect()),
            DType::F64 => CpuBuffer::F64(as_f64),
            DType::F16 => CpuBuffer::F16(as_f64.into_iter().map(|x| F16::from_f32(x as f32)).collect()),
            DType::BF16 => CpuBuffer::BF16(as_f64.into_iter().map(|x| BF16::from_f32(x as f32)).collect()),
            DType::I32 => CpuBuffer::I32(as_f64.into_iter().map(|x| x as i32).collect()),
            DType::I64 => CpuBuffer::I64(as_f64.into_iter().map(|x| x as i64).collect()),
        };
        Ok(Self::from_buffer(buf))
    }
}

impl Storage for CpuStorage {
    fn dtype(&self) -> DType {
        match self.buffer() {
            CpuBuffer::F32(_) => DType::F32,
            CpuBuffer::F64(_) => DType::F64,
            CpuBuffer::F16(_) => DType::F16,
            CpuBuffer::BF16(_) => DType::BF16,
            CpuBuffer::I32(_) => DType::I32,
            CpuBuffer::I64(_) => DType::I64,
        }
    }

    fn len(&self) -> usize {
        match self.buffer() {
            CpuBuffer::F32(v) => v.len(),
            CpuBuffer::F64(v) => v.len(),
            CpuBuffer::F16(v) => v.len(),
            CpuBuffer::BF16(v) => v.len(),
            CpuBuffer::I32(v) => v.len(),
            CpuBuffer::I64(v) => v.len(),
        }
    }

    fn device(&self) -> Device {
        Device::Cpu
    }
}
