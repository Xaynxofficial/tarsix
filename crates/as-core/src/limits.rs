//! Security and resource-control limits for untrusted inputs and allocations.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Maximum supported tensor rank (number of dimensions).
///
/// Bounding rank prevents stack exhaustion or exponential stride/broadcast blowups
/// when parsing untrusted model headers or tensor shapes.
pub const MAX_RANK: usize = 16;

/// Default maximum single-tensor allocation size (4 GiB on 64-bit, 512 MiB on 32-bit/WASM).
#[cfg(target_pointer_width = "64")]
pub const DEFAULT_MAX_TENSOR_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Default maximum single-tensor allocation size on 32-bit targets such as WebAssembly (512 MiB).
#[cfg(not(target_pointer_width = "64"))]
pub const DEFAULT_MAX_TENSOR_BYTES: usize = 512 * 1024 * 1024;

/// Configurable resource limits for safe memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLimits {
    /// Maximum byte size permitted for any single tensor buffer.
    pub max_tensor_bytes: usize,
    /// Maximum permitted tensor rank (capped at [`MAX_RANK`]).
    pub max_rank: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
            max_rank: MAX_RANK,
        }
    }
}

impl MemoryLimits {
    /// Creates a new [`MemoryLimits`] with explicit byte and rank ceilings.
    pub fn new(max_tensor_bytes: usize, max_rank: usize) -> Self {
        Self {
            max_tensor_bytes,
            max_rank: max_rank.min(MAX_RANK),
        }
    }

    /// Validates that allocating `numel` elements of `dtype` does not overflow
    /// `usize` and stays within `max_tensor_bytes`.
    pub fn check_allocation(&self, numel: usize, dtype: DType) -> Result<usize> {
        let bytes = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or(Error::ArithmeticOverflow {
                context: "tensor byte size calculation",
            })?;
        if bytes > self.max_tensor_bytes {
            return Err(Error::AllocationLimitExceeded {
                requested_bytes: bytes,
                limit_bytes: self.max_tensor_bytes,
            });
        }
        Ok(bytes)
    }

    /// Validates that a rank does not exceed the configured ceiling.
    pub fn check_rank(&self, rank: usize, context: &'static str) -> Result<()> {
        if rank > self.max_rank {
            return Err(Error::InvalidRank {
                context,
                rank,
                expected: self.max_rank,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_allocation_above_limit() {
        let limits = MemoryLimits::new(1024, 8);
        let err = limits.check_allocation(257, DType::F32).unwrap_err();
        assert_eq!(
            err,
            Error::AllocationLimitExceeded {
                requested_bytes: 1028,
                limit_bytes: 1024,
            }
        );
    }

    #[test]
    fn rejects_byte_size_overflow() {
        let limits = MemoryLimits::default();
        let err = limits.check_allocation(usize::MAX, DType::F64).unwrap_err();
        assert_eq!(
            err,
            Error::ArithmeticOverflow {
                context: "tensor byte size calculation",
            }
        );
    }
}
