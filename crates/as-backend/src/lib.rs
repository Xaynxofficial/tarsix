//! Hardware-independent execution backend and storage contracts.
//!
//! All compute backends (CPU, future Metal, CUDA, WebGPU) implement [`Backend`]
//! and [`Storage`], maintaining strict architectural separation between
//! high-level tensor semantics and hardware kernel dispatch.

use as_core::{BF16, DType, Device, DeviceCapabilities, F16, Layout, MemoryLimits, Result, Shape};

/// Element-wise binary arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Element-wise addition (`lhs + rhs`).
    Add,
    /// Element-wise subtraction (`lhs - rhs`).
    Sub,
    /// Element-wise multiplication (`lhs * rhs`).
    Mul,
    /// Element-wise division (`lhs / rhs`).
    Div,
}

/// Element-wise unary mathematical operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    /// Arithmetic negation (`-x`).
    Neg,
    /// Absolute value (`|x|`).
    Abs,
    /// Natural exponential (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Ln,
    /// Square root (`sqrt(x)`).
    Sqrt,
    /// Rectified linear unit (`max(0, x)`).
    Relu,
    /// Raise to floating-point exponent (`x^p`).
    Powf(f64),
}

/// Reduction operations over all elements or along a single axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Arithmetic sum.
    Sum,
    /// Arithmetic mean (floating-point dtypes only).
    Mean,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
}

/// Backend-managed contiguous memory buffer backing one or more tensor views.
pub trait Storage: Clone + Send + Sync + 'static {
    /// Returns the stored element [`DType`].
    fn dtype(&self) -> DType;

    /// Returns the total physical number of elements in this buffer.
    fn len(&self) -> usize;

    /// Returns `true` if the buffer contains zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`Device`] where this buffer resides.
    fn device(&self) -> Device;
}

/// Hardware-independent execution engine contract.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Concrete storage type managed by this backend.
    type Storage: Storage;

    /// Returns hardware capabilities for this backend instance.
    fn capabilities(&self) -> DeviceCapabilities;

    /// Returns the security and resource limits enforced by this backend.
    fn limits(&self) -> MemoryLimits;

    /// Allocates a zero-initialized storage buffer of `numel` elements.
    fn zeros(&self, numel: usize, dtype: DType) -> Result<Self::Storage>;

    /// Allocates a one-initialized storage buffer of `numel` elements.
    fn ones(&self, numel: usize, dtype: DType) -> Result<Self::Storage>;

    /// Allocates a buffer filled with `value` converted to `dtype`.
    fn full_f64(&self, numel: usize, value: f64, dtype: DType) -> Result<Self::Storage>;

    /// Creates storage from a host `f32` slice.
    fn from_f32_slice(&self, data: &[f32]) -> Result<Self::Storage>;

    /// Creates storage from a host `f64` slice.
    fn from_f64_slice(&self, data: &[f64]) -> Result<Self::Storage>;

    /// Creates storage from a host `F16` slice.
    fn from_f16_slice(&self, data: &[F16]) -> Result<Self::Storage>;

    /// Creates storage from a host `BF16` slice.
    fn from_bf16_slice(&self, data: &[BF16]) -> Result<Self::Storage>;

    /// Creates storage from a host `i32` slice.
    fn from_i32_slice(&self, data: &[i32]) -> Result<Self::Storage>;

    /// Creates storage from a host `i64` slice.
    fn from_i64_slice(&self, data: &[i64]) -> Result<Self::Storage>;

    /// Copies elements in logical layout order to a host `Vec<f32>`.
    fn to_f32_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<f32>>;

    /// Copies elements in logical layout order to a host `Vec<f64>`.
    fn to_f64_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<f64>>;

    /// Copies elements in logical layout order to a host `Vec<F16>`.
    fn to_f16_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<F16>>;

    /// Copies elements in logical layout order to a host `Vec<BF16>`.
    fn to_bf16_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<BF16>>;

    /// Copies elements in logical layout order to a host `Vec<i32>`.
    fn to_i32_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<i32>>;

    /// Copies elements in logical layout order to a host `Vec<i64>`.
    fn to_i64_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<i64>>;

    /// Materializes a potentially strided or offset view into a contiguous buffer.
    fn contiguous(&self, storage: &Self::Storage, layout: &Layout) -> Result<Self::Storage>;

    /// Casts elements from `storage` to `target_dtype`, returning a contiguous buffer.
    fn cast(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        target_dtype: DType,
    ) -> Result<Self::Storage>;

    /// Executes an element-wise binary operation over two views with identical
    /// (already broadcasted) shapes.
    fn binary_op(
        &self,
        op: BinaryOp,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage>;

    /// Executes an element-wise unary operation over a view.
    fn unary_op(
        &self,
        op: UnaryOp,
        src: &Self::Storage,
        layout: &Layout,
    ) -> Result<Self::Storage>;

    /// Executes 2D or batched N-D matrix multiplication with broadcastable batch dimensions.
    fn matmul(
        &self,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Shape)>;

    /// Executes a global or axis-wise reduction.
    fn reduce(
        &self,
        op: ReduceOp,
        src: &Self::Storage,
        layout: &Layout,
        axis: Option<usize>,
        keepdim: bool,
    ) -> Result<(Self::Storage, Shape)>;
}
