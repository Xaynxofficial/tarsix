//! Strongly typed error hierarchy for all engine operations.

use core::fmt;

use crate::dtype::DType;

/// Convenient result alias using [`Error`].
pub type Result<T> = core::result::Result<T, Error>;

/// Comprehensive error enumeration covering validation, numerical, layout,
/// hardware, and security/resource-limit failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two or more shapes are incompatible for the requested operation.
    ShapeMismatch {
        /// Human-readable operation context.
        context: &'static str,
        /// Expected or left-hand shape.
        expected: Vec<usize>,
        /// Actual or right-hand shape.
        actual: Vec<usize>,
    },
    /// Shapes cannot be broadcast according to right-aligned broadcasting rules.
    BroadcastIncompatible {
        /// Left-hand shape.
        lhs: Vec<usize>,
        /// Right-hand shape.
        rhs: Vec<usize>,
    },
    /// Tensor rank exceeds [`crate::MAX_RANK`] or violates operation rank constraints.
    InvalidRank {
        /// Operation or constraint context.
        context: &'static str,
        /// Observed rank.
        rank: usize,
        /// Maximum or expected rank.
        expected: usize,
    },
    /// An axis index is out of bounds for the tensor's rank.
    InvalidAxis {
        /// Requested axis.
        axis: usize,
        /// Tensor rank.
        rank: usize,
    },
    /// Duplicate or missing axes in a permutation specification.
    InvalidPermutation {
        /// Provided permutation.
        perm: Vec<usize>,
        /// Tensor rank.
        rank: usize,
    },
    /// Slice bounds or step are invalid for the target dimension.
    InvalidSlice {
        /// Axis being sliced.
        axis: usize,
        /// Dimension length along that axis.
        dim: usize,
        /// Requested start index.
        start: usize,
        /// Requested end index (exclusive).
        end: usize,
        /// Requested step.
        step: usize,
    },
    /// Data types do not match where homogeneous dtypes are required.
    DTypeMismatch {
        /// Expected dtype.
        expected: DType,
        /// Actual dtype.
        actual: DType,
    },
    /// Operation is not supported for the given dtype (e.g. `exp` on `I32`).
    UnsupportedDType {
        /// Operation name.
        op: &'static str,
        /// Unsupported dtype.
        dtype: DType,
    },
    /// Devices do not match where co-located tensors are required.
    DeviceMismatch {
        /// Expected device description.
        expected: String,
        /// Actual device description.
        actual: String,
    },
    /// Requested operation requires a contiguous memory layout.
    NonContiguousLayout {
        /// Operation context.
        context: &'static str,
    },
    /// Integer overflow detected during shape, stride, or byte-size computation.
    ArithmeticOverflow {
        /// Operation context.
        context: &'static str,
    },
    /// Requested allocation exceeds configured security [`crate::MemoryLimits`].
    AllocationLimitExceeded {
        /// Requested byte count.
        requested_bytes: usize,
        /// Configured maximum bytes.
        limit_bytes: usize,
    },
    /// Buffer length does not match expected element count.
    BufferLengthMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
    /// Index out of bounds during validated element access.
    IndexOutOfBounds {
        /// Requested index.
        index: usize,
        /// Valid length.
        len: usize,
    },
    /// Invalid numerical argument (e.g. division by zero or invalid range).
    InvalidArgument {
        /// Parameter or context description.
        context: &'static str,
        /// Detailed reason.
        reason: String,
    },
    /// Internal invariant violation (defensive fallback instead of panicking).
    InternalInvariant {
        /// Invariant description.
        message: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch in {context}: expected {expected:?}, got {actual:?}"
            ),
            Self::BroadcastIncompatible { lhs, rhs } => {
                write!(f, "shapes {lhs:?} and {rhs:?} are not broadcast-compatible")
            }
            Self::InvalidRank {
                context,
                rank,
                expected,
            } => write!(
                f,
                "invalid rank in {context}: got {rank}, expected/max {expected}"
            ),
            Self::InvalidAxis { axis, rank } => {
                write!(f, "axis {axis} is out of bounds for tensor of rank {rank}")
            }
            Self::InvalidPermutation { perm, rank } => {
                write!(f, "invalid permutation {perm:?} for rank {rank}")
            }
            Self::InvalidSlice {
                axis,
                dim,
                start,
                end,
                step,
            } => write!(
                f,
                "invalid slice [{start}..{end}:{step}] on axis {axis} of length {dim}"
            ),
            Self::DTypeMismatch { expected, actual } => {
                write!(f, "dtype mismatch: expected {expected}, got {actual}")
            }
            Self::UnsupportedDType { op, dtype } => {
                write!(f, "operation `{op}` is not supported for dtype {dtype}")
            }
            Self::DeviceMismatch { expected, actual } => {
                write!(f, "device mismatch: expected {expected}, got {actual}")
            }
            Self::NonContiguousLayout { context } => {
                write!(f, "operation `{context}` requires a contiguous tensor layout")
            }
            Self::ArithmeticOverflow { context } => {
                write!(f, "integer overflow during `{context}`")
            }
            Self::AllocationLimitExceeded {
                requested_bytes,
                limit_bytes,
            } => write!(
                f,
                "allocation of {requested_bytes} bytes exceeds security limit of {limit_bytes} bytes"
            ),
            Self::BufferLengthMismatch { expected, actual } => write!(
                f,
                "buffer length mismatch: expected {expected} elements, got {actual}"
            ),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
            Self::InvalidArgument { context, reason } => {
                write!(f, "invalid argument in `{context}`: {reason}")
            }
            Self::InternalInvariant { message } => {
                write!(f, "internal invariant violated: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}
