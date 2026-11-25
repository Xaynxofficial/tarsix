//! Safe, deterministic, cache-blocked CPU backend implementation.

pub mod ops;
pub mod storage;

use as_backend::{Backend, BinaryOp, ReduceOp, Storage, UnaryOp};
use as_core::{BF16, DType, Device, DeviceCapabilities, Error, F16, Layout, MemoryLimits, Result, Shape};

pub use storage::{CpuBuffer, CpuStorage};

/// Reference CPU execution backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend {
    limits: MemoryLimits,
}

impl CpuBackend {
    /// Creates a [`CpuBackend`] with default [`MemoryLimits`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`CpuBackend`] enforcing custom [`MemoryLimits`].
    pub fn with_limits(limits: MemoryLimits) -> Self {
        Self { limits }
    }
}

impl Backend for CpuBackend {
    type Storage = CpuStorage;

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            device: Device::Cpu,
            name: "as-cpu (safe deterministic reference)".to_owned(),
            native_f16: false,
            native_bf16: false,
            deterministic: true,
        }
    }

    fn limits(&self) -> MemoryLimits {
        self.limits
    }

    fn zeros(&self, numel: usize, dtype: DType) -> Result<Self::Storage> {
        CpuStorage::full_f64(numel, 0.0, dtype, self.limits)
    }

    fn ones(&self, numel: usize, dtype: DType) -> Result<Self::Storage> {
        CpuStorage::full_f64(numel, 1.0, dtype, self.limits)
    }

    fn full_f64(&self, numel: usize, value: f64, dtype: DType) -> Result<Self::Storage> {
        CpuStorage::full_f64(numel, value, dtype, self.limits)
    }

    fn from_f32_slice(&self, data: &[f32]) -> Result<Self::Storage> {
        self.limits.check_allocation(data.len(), DType::F32)?;
        Ok(CpuStorage::from_buffer(CpuBuffer::F32(data.to_vec())))
    }

    fn from_f64_slice(&self, data: &[f64]) -> Result<Self::Storage> {
        self.limits.check_allocation(data.len(), DType::F64)?;
        Ok(CpuStorage::from_buffer(CpuBuffer::F64(data.to_vec())))
    }

    fn from_f16_slice(&self, data: &[F16]) -> Result<Self::Storage> {
        self.limits.check_allocation(data.len(), DType::F16)?;
        Ok(CpuStorage::from_buffer(CpuBuffer::F16(data.to_vec())))
    }

    fn from_bf16_slice(&self, data: &[BF16]) -> Result<Self::Storage> {
        self.limits.check_allocation(data.len(), DType::BF16)?;
        Ok(CpuStorage::from_buffer(CpuBuffer::BF16(data.to_vec())))
    }

    fn from_i32_slice(&self, data: &[i32]) -> Result<Self::Storage> {
        self.limits.check_allocation(data.len(), DType::I32)?;
        Ok(CpuStorage::from_buffer(CpuBuffer::I32(data.to_vec())))
    }

    fn from_i64_slice(&self, data: &[i64]) -> Result<Self::Storage> {
        self.limits.check_allocation(data.len(), DType::I64)?;
        Ok(CpuStorage::from_buffer(CpuBuffer::I64(data.to_vec())))
    }

    fn to_f32_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<f32>> {
        match storage.buffer() {
            CpuBuffer::F32(v) => CpuStorage::materialize_slice(v, layout, self.limits, DType::F32),
            _ => Err(Error::DTypeMismatch {
                expected: DType::F32,
                actual: storage.dtype(),
            }),
        }
    }

    fn to_f64_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<f64>> {
        match storage.buffer() {
            CpuBuffer::F64(v) => CpuStorage::materialize_slice(v, layout, self.limits, DType::F64),
            _ => Err(Error::DTypeMismatch {
                expected: DType::F64,
                actual: storage.dtype(),
            }),
        }
    }

    fn to_f16_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<F16>> {
        match storage.buffer() {
            CpuBuffer::F16(v) => CpuStorage::materialize_slice(v, layout, self.limits, DType::F16),
            _ => Err(Error::DTypeMismatch {
                expected: DType::F16,
                actual: storage.dtype(),
            }),
        }
    }

    fn to_bf16_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<BF16>> {
        match storage.buffer() {
            CpuBuffer::BF16(v) => CpuStorage::materialize_slice(v, layout, self.limits, DType::BF16),
            _ => Err(Error::DTypeMismatch {
                expected: DType::BF16,
                actual: storage.dtype(),
            }),
        }
    }

    fn to_i32_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<i32>> {
        match storage.buffer() {
            CpuBuffer::I32(v) => CpuStorage::materialize_slice(v, layout, self.limits, DType::I32),
            _ => Err(Error::DTypeMismatch {
                expected: DType::I32,
                actual: storage.dtype(),
            }),
        }
    }

    fn to_i64_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<i64>> {
        match storage.buffer() {
            CpuBuffer::I64(v) => CpuStorage::materialize_slice(v, layout, self.limits, DType::I64),
            _ => Err(Error::DTypeMismatch {
                expected: DType::I64,
                actual: storage.dtype(),
            }),
        }
    }

    fn contiguous(&self, storage: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        storage.contiguous(layout, self.limits)
    }

    fn cast(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        target_dtype: DType,
    ) -> Result<Self::Storage> {
        storage.cast(layout, target_dtype, self.limits)
    }

    fn binary_op(
        &self,
        op: BinaryOp,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage> {
        ops::elementwise::binary_op(op, lhs, lhs_layout, rhs, rhs_layout, self.limits)
    }

    fn unary_op(
        &self,
        op: UnaryOp,
        src: &Self::Storage,
        layout: &Layout,
    ) -> Result<Self::Storage> {
        ops::elementwise::unary_op(op, src, layout, self.limits)
    }

    fn matmul(
        &self,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Shape)> {
        ops::matmul::matmul(lhs, lhs_layout, rhs, rhs_layout, self.limits)
    }

    fn reduce(
        &self,
        op: ReduceOp,
        src: &Self::Storage,
        layout: &Layout,
        axis: Option<usize>,
        keepdim: bool,
    ) -> Result<(Self::Storage, Shape)> {
        ops::reduce::reduce(op, src, layout, axis, keepdim, self.limits)
    }
}
