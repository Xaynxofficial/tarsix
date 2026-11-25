//! Public N-dimensional Tensor API for the independent AI computing infrastructure.
//!
//! [`Tensor`] is generic over any [`as_backend::Backend`] and defaults to [`as_cpu::CpuBackend`],
//! providing zero-copy view algebra, broadcast-aware arithmetic, tiled matrix multiplication,
//! reductions, and strict error-checked memory safety.

use core::fmt;
use core::ops::{Add, Div, Mul, Sub};

pub use as_backend::{Backend, BinaryOp, ReduceOp, Storage, UnaryOp};
pub use as_core::{
    BF16, DType, Device, DeviceCapabilities, Error, F16, Layout, MAX_RANK, MemoryLimits, Result,
    Shape, SliceRange, TensorElement,
};
pub use as_cpu::CpuBackend;

/// N-dimensional tensor backed by a hardware-independent [`Backend`].
#[derive(Clone)]
pub struct Tensor<B: Backend = CpuBackend> {
    storage: B::Storage,
    layout: Layout,
    backend: B,
}

impl Tensor<CpuBackend> {
    /// Creates a zero-filled CPU tensor with default [`MemoryLimits`].
    pub fn zeros(shape: impl Into<Vec<usize>>, dtype: DType) -> Result<Self> {
        Self::zeros_on(shape, dtype, CpuBackend::new())
    }

    /// Creates a one-filled CPU tensor with default [`MemoryLimits`].
    pub fn ones(shape: impl Into<Vec<usize>>, dtype: DType) -> Result<Self> {
        Self::ones_on(shape, dtype, CpuBackend::new())
    }

    /// Creates a CPU tensor filled with `value`.
    pub fn full(shape: impl Into<Vec<usize>>, value: f64, dtype: DType) -> Result<Self> {
        Self::full_on(shape, value, dtype, CpuBackend::new())
    }

    /// Creates a CPU tensor from a contiguous `f32` slice.
    pub fn from_f32(data: &[f32], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_f32_on(data, shape, CpuBackend::new())
    }

    /// Creates a CPU tensor from a contiguous `f64` slice.
    pub fn from_f64(data: &[f64], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_f64_on(data, shape, CpuBackend::new())
    }

    /// Creates a CPU tensor from a contiguous [`F16`] slice.
    pub fn from_f16(data: &[F16], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_f16_on(data, shape, CpuBackend::new())
    }

    /// Creates a CPU tensor from a contiguous [`BF16`] slice.
    pub fn from_bf16(data: &[BF16], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_bf16_on(data, shape, CpuBackend::new())
    }

    /// Creates a CPU tensor from a contiguous `i32` slice.
    pub fn from_i32(data: &[i32], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_i32_on(data, shape, CpuBackend::new())
    }

    /// Creates a CPU tensor from a contiguous `i64` slice.
    pub fn from_i64(data: &[i64], shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_i64_on(data, shape, CpuBackend::new())
    }

    /// Creates a 1-D CPU tensor with evenly spaced values `[start, start + step, ...) < end`.
    pub fn arange_f32(start: f32, end: f32, step: f32) -> Result<Self> {
        if !step.is_finite() || step == 0.0 {
            return Err(Error::InvalidArgument {
                context: "Tensor::arange_f32",
                reason: "step must be finite and non-zero".to_owned(),
            });
        }
        let mut data = Vec::new();
        let mut current = start;
        if step > 0.0 {
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            while current > end {
                data.push(current);
                current += step;
            }
        }
        let len = data.len();
        Self::from_f32(&data, vec![len])
    }

    /// Creates a 2-D identity matrix of size `n x n` on CPU.
    pub fn eye(n: usize, dtype: DType) -> Result<Self> {
        let shape = Shape::new(vec![n, n])?;
        let backend = CpuBackend::new();
        backend.limits().check_allocation(shape.numel(), dtype)?;
        let mut vals = vec![0.0f64; shape.numel()];
        for i in 0..n {
            if let Some(slot) = vals.get_mut(i * n + i) {
                *slot = 1.0;
            }
        }
        let base = Self::from_f64_on(&vals, vec![n, n], backend)?;
        base.to_dtype(dtype)
    }
}

impl<B: Backend> Tensor<B> {
    /// Creates a zero-filled tensor on the specified `backend`.
    pub fn zeros_on(shape: impl Into<Vec<usize>>, dtype: DType, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        backend.limits().check_rank(shape.rank(), "Tensor::zeros_on")?;
        let storage = backend.zeros(shape.numel(), dtype)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a one-filled tensor on the specified `backend`.
    pub fn ones_on(shape: impl Into<Vec<usize>>, dtype: DType, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        backend.limits().check_rank(shape.rank(), "Tensor::ones_on")?;
        let storage = backend.ones(shape.numel(), dtype)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a constant-filled tensor on the specified `backend`.
    pub fn full_on(
        shape: impl Into<Vec<usize>>,
        value: f64,
        dtype: DType,
        backend: B,
    ) -> Result<Self> {
        let shape = Shape::new(shape)?;
        backend.limits().check_rank(shape.rank(), "Tensor::full_on")?;
        let storage = backend.full_f64(shape.numel(), value, dtype)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a tensor from an `f32` slice on `backend`.
    pub fn from_f32_on(data: &[f32], shape: impl Into<Vec<usize>>, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        validate_slice_len(data.len(), shape.numel())?;
        let storage = backend.from_f32_slice(data)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a tensor from an `f64` slice on `backend`.
    pub fn from_f64_on(data: &[f64], shape: impl Into<Vec<usize>>, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        validate_slice_len(data.len(), shape.numel())?;
        let storage = backend.from_f64_slice(data)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a tensor from an [`F16`] slice on `backend`.
    pub fn from_f16_on(data: &[F16], shape: impl Into<Vec<usize>>, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        validate_slice_len(data.len(), shape.numel())?;
        let storage = backend.from_f16_slice(data)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a tensor from a [`BF16`] slice on `backend`.
    pub fn from_bf16_on(data: &[BF16], shape: impl Into<Vec<usize>>, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        validate_slice_len(data.len(), shape.numel())?;
        let storage = backend.from_bf16_slice(data)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a tensor from an `i32` slice on `backend`.
    pub fn from_i32_on(data: &[i32], shape: impl Into<Vec<usize>>, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        validate_slice_len(data.len(), shape.numel())?;
        let storage = backend.from_i32_slice(data)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    /// Creates a tensor from an `i64` slice on `backend`.
    pub fn from_i64_on(data: &[i64], shape: impl Into<Vec<usize>>, backend: B) -> Result<Self> {
        let shape = Shape::new(shape)?;
        validate_slice_len(data.len(), shape.numel())?;
        let storage = backend.from_i64_slice(data)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(shape),
            backend,
        })
    }

    // --- Inspection ---

    /// Returns the tensor's [`Shape`].
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Returns the tensor's dimension slice.
    pub fn dims(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    /// Returns the tensor's strides.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    /// Returns the tensor's [`Layout`].
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the tensor's [`DType`].
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Returns the tensor's [`Device`].
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// Returns the total number of logical elements.
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Returns `true` if the tensor is stored in C-contiguous order.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns a reference to the underlying backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    // --- Host Extraction ---

    /// Copies tensor values to a `Vec<f32>` in logical order.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        self.backend.to_f32_vec(&self.storage, &self.layout)
    }

    /// Copies tensor values to a `Vec<f64>` in logical order.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        self.backend.to_f64_vec(&self.storage, &self.layout)
    }

    /// Copies tensor values to a `Vec<F16>` in logical order.
    pub fn to_f16_vec(&self) -> Result<Vec<F16>> {
        self.backend.to_f16_vec(&self.storage, &self.layout)
    }

    /// Copies tensor values to a `Vec<BF16>` in logical order.
    pub fn to_bf16_vec(&self) -> Result<Vec<BF16>> {
        self.backend.to_bf16_vec(&self.storage, &self.layout)
    }

    /// Copies tensor values to a `Vec<i32>` in logical order.
    pub fn to_i32_vec(&self) -> Result<Vec<i32>> {
        self.backend.to_i32_vec(&self.storage, &self.layout)
    }

    /// Copies tensor values to a `Vec<i64>` in logical order.
    pub fn to_i64_vec(&self) -> Result<Vec<i64>> {
        self.backend.to_i64_vec(&self.storage, &self.layout)
    }

    // --- Zero-Copy Layout & View Operations ---

    /// Returns a contiguous tensor (zero-copy if already contiguous at offset 0).
    pub fn contiguous(&self) -> Result<Self> {
        let storage = self.backend.contiguous(&self.storage, &self.layout)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(self.shape().clone()),
            backend: self.backend.clone(),
        })
    }

    /// Casts this tensor to `target_dtype`.
    pub fn to_dtype(&self, target_dtype: DType) -> Result<Self> {
        let storage = self.backend.cast(&self.storage, &self.layout, target_dtype)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(self.shape().clone()),
            backend: self.backend.clone(),
        })
    }

    /// Reshapes the tensor to `new_shape`.
    ///
    /// Zero-copy if contiguous; materializes a contiguous copy first if non-contiguous.
    pub fn reshape(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let new_shape = Shape::new(new_shape)?;
        if self.is_contiguous() {
            let layout = self.layout.reshape(new_shape)?;
            Ok(Self {
                storage: self.storage.clone(),
                layout,
                backend: self.backend.clone(),
            })
        } else {
            let contig = self.contiguous()?;
            let layout = contig.layout.reshape(new_shape)?;
            Ok(Self {
                storage: contig.storage,
                layout,
                backend: self.backend.clone(),
            })
        }
    }

    /// Swaps two dimensions (zero-copy).
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.layout.transpose(axis_a, axis_b)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout,
            backend: self.backend.clone(),
        })
    }

    /// Reorders dimensions according to `perm` (zero-copy).
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(perm)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout,
            backend: self.backend.clone(),
        })
    }

    /// Inserts a size-1 dimension at `axis` (zero-copy).
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        let layout = self.layout.unsqueeze(axis)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout,
            backend: self.backend.clone(),
        })
    }

    /// Removes a size-1 dimension at `axis` (zero-copy).
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        let layout = self.layout.squeeze(axis)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout,
            backend: self.backend.clone(),
        })
    }

    /// Broadcasts this tensor to `target_shape` without copying data (zero-stride view).
    pub fn broadcast_to(&self, target_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let target = Shape::new(target_shape)?;
        let layout = self.layout.broadcast_to(&target)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout,
            backend: self.backend.clone(),
        })
    }

    /// Returns a zero-copy sliced view along the leading axes specified by `ranges`.
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<Self> {
        let layout = self.layout.slice(ranges)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout,
            backend: self.backend.clone(),
        })
    }

    // --- Arithmetic Operations (with automatic broadcasting) ---

    fn binary_broadcast(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        if self.device() != other.device() {
            return Err(Error::DeviceMismatch {
                expected: self.device().to_string(),
                actual: other.device().to_string(),
            });
        }
        let out_shape = self.shape().broadcast_shape(other.shape())?;
        let lhs_layout = self.layout.broadcast_to(&out_shape)?;
        let rhs_layout = other.layout.broadcast_to(&out_shape)?;
        let storage = self.backend.binary_op(
            op,
            &self.storage,
            &lhs_layout,
            &other.storage,
            &rhs_layout,
        )?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(out_shape),
            backend: self.backend.clone(),
        })
    }

    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.binary_broadcast(other, BinaryOp::Add)
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.binary_broadcast(other, BinaryOp::Sub)
    }

    /// Element-wise multiplication with broadcasting.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.binary_broadcast(other, BinaryOp::Mul)
    }

    /// Element-wise division with broadcasting.
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary_broadcast(other, BinaryOp::Div)
    }

    fn unary(&self, op: UnaryOp) -> Result<Self> {
        let storage = self.backend.unary_op(op, &self.storage, &self.layout)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(self.shape().clone()),
            backend: self.backend.clone(),
        })
    }

    /// Element-wise negation (`-x`).
    pub fn neg(&self) -> Result<Self> {
        self.unary(UnaryOp::Neg)
    }

    /// Element-wise absolute value (`|x|`).
    pub fn abs(&self) -> Result<Self> {
        self.unary(UnaryOp::Abs)
    }

    /// Element-wise natural exponential (`e^x`).
    pub fn exp(&self) -> Result<Self> {
        self.unary(UnaryOp::Exp)
    }

    /// Element-wise natural logarithm (`ln(x)`).
    pub fn ln(&self) -> Result<Self> {
        self.unary(UnaryOp::Ln)
    }

    /// Element-wise square root (`sqrt(x)`).
    pub fn sqrt(&self) -> Result<Self> {
        self.unary(UnaryOp::Sqrt)
    }

    /// Element-wise ReLU (`max(0, x)`).
    pub fn relu(&self) -> Result<Self> {
        self.unary(UnaryOp::Relu)
    }

    /// Element-wise power (`x^exponent`).
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        self.unary(UnaryOp::Powf(exponent))
    }

    // --- Matrix Multiplication ---

    /// 2D or batched N-D matrix multiplication with broadcastable batch dimensions.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch {
                expected: self.device().to_string(),
                actual: rhs.device().to_string(),
            });
        }
        let (storage, out_shape) =
            self.backend
                .matmul(&self.storage, &self.layout, &rhs.storage, &rhs.layout)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(out_shape),
            backend: self.backend.clone(),
        })
    }

    // --- Reductions ---

    /// Reduces all elements to a scalar sum.
    pub fn sum(&self) -> Result<Self> {
        self.reduce_impl(ReduceOp::Sum, None, false)
    }

    /// Sums elements along `axis`.
    pub fn sum_axis(&self, axis: usize, keepdim: bool) -> Result<Self> {
        self.reduce_impl(ReduceOp::Sum, Some(axis), keepdim)
    }

    /// Reduces all elements to a scalar mean.
    pub fn mean(&self) -> Result<Self> {
        self.reduce_impl(ReduceOp::Mean, None, false)
    }

    /// Computes mean along `axis`.
    pub fn mean_axis(&self, axis: usize, keepdim: bool) -> Result<Self> {
        self.reduce_impl(ReduceOp::Mean, Some(axis), keepdim)
    }

    /// Computes global maximum.
    pub fn max(&self) -> Result<Self> {
        self.reduce_impl(ReduceOp::Max, None, false)
    }

    /// Computes maximum along `axis`.
    pub fn max_axis(&self, axis: usize, keepdim: bool) -> Result<Self> {
        self.reduce_impl(ReduceOp::Max, Some(axis), keepdim)
    }

    /// Computes global minimum.
    pub fn min(&self) -> Result<Self> {
        self.reduce_impl(ReduceOp::Min, None, false)
    }

    /// Computes minimum along `axis`.
    pub fn min_axis(&self, axis: usize, keepdim: bool) -> Result<Self> {
        self.reduce_impl(ReduceOp::Min, Some(axis), keepdim)
    }

    fn reduce_impl(&self, op: ReduceOp, axis: Option<usize>, keepdim: bool) -> Result<Self> {
        let (storage, out_shape) =
            self.backend
                .reduce(op, &self.storage, &self.layout, axis, keepdim)?;
        Ok(Self {
            storage,
            layout: Layout::contiguous(out_shape),
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", self.shape())
            .field("strides", &self.strides())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}

impl<B: Backend> Add for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

impl<B: Backend> Sub for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

impl<B: Backend> Mul for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

impl<B: Backend> Div for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(self, rhs)
    }
}

fn validate_slice_len(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::BufferLengthMismatch { expected, actual });
    }
    Ok(())
}
