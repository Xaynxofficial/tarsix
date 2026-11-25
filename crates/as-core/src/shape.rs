//! N-dimensional [`Shape`], [`Layout`] (strides & offset), broadcasting,
//! and zero-copy view transformations.
//!
//! Every arithmetic calculation uses overflow-checked operations to defend
//! against crafted or corrupted tensor metadata.

use core::fmt;

use crate::error::{Error, Result};
use crate::limits::MAX_RANK;

/// Dimension specification for an N-dimensional tensor.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a validated [`Shape`], checking [`MAX_RANK`] and overflow of total element count.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.len() > MAX_RANK {
            return Err(Error::InvalidRank {
                context: "Shape::new",
                rank: dims.len(),
                expected: MAX_RANK,
            });
        }
        // Validate that element count does not overflow usize.
        let mut total: usize = 1;
        for &d in &dims {
            total = total.checked_mul(d).ok_or(Error::ArithmeticOverflow {
                context: "Shape::numel",
            })?;
        }
        Ok(Self { dims })
    }

    /// Creates a scalar (0-D) shape.
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Returns the dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of logical elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().copied().product()
    }

    /// Returns the size of dimension `axis`, or an [`Error::InvalidAxis`] if out of bounds.
    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.dims
            .get(axis)
            .copied()
            .ok_or(Error::InvalidAxis {
                axis,
                rank: self.dims.len(),
            })
    }

    /// Computes standard row-major (C-contiguous) strides for this shape.
    pub fn contiguous_strides(&self) -> Vec<usize> {
        let rank = self.dims.len();
        if rank == 0 {
            return Vec::new();
        }
        let mut strides = vec![1usize; rank];
        let mut acc = 1usize;
        for i in (0..rank).rev() {
            if let (Some(s), Some(&d)) = (strides.get_mut(i), self.dims.get(i)) {
                *s = acc;
                // Safe: Shape::new already verified product does not overflow.
                acc = acc.saturating_mul(d);
            }
        }
        strides
    }

    /// Computes the broadcasted output shape of two shapes using right-aligned
    /// NumPy/PyTorch broadcasting rules.
    pub fn broadcast_shape(&self, other: &Self) -> Result<Self> {
        let rank = self.rank().max(other.rank());
        if rank > MAX_RANK {
            return Err(Error::InvalidRank {
                context: "Shape::broadcast_shape",
                rank,
                expected: MAX_RANK,
            });
        }
        let mut out = vec![0usize; rank];
        for i in 0..rank {
            let lhs_dim = if i < self.rank() {
                self.dims.get(self.rank() - 1 - i).copied().unwrap_or(1)
            } else {
                1
            };
            let rhs_dim = if i < other.rank() {
                other.dims.get(other.rank() - 1 - i).copied().unwrap_or(1)
            } else {
                1
            };

            let dim = if lhs_dim == rhs_dim {
                lhs_dim
            } else if lhs_dim == 1 {
                rhs_dim
            } else if rhs_dim == 1 {
                lhs_dim
            } else {
                return Err(Error::BroadcastIncompatible {
                    lhs: self.dims.clone(),
                    rhs: other.dims.clone(),
                });
            };

            if let Some(slot) = out.get_mut(rank - 1 - i) {
                *slot = dim;
            }
        }
        Self::new(out)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

/// Half-open slice specification `[start..end]` with a positive `step` along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceRange {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index (`None` means end of dimension).
    pub end: Option<usize>,
    /// Positive stride step (defaults to 1).
    pub step: usize,
}

impl SliceRange {
    /// Full axis slice (`..`).
    pub const fn full() -> Self {
        Self {
            start: 0,
            end: None,
            step: 1,
        }
    }

    /// Creates a slice `[start..end)` with step 1.
    pub const fn range(start: usize, end: usize) -> Self {
        Self {
            start,
            end: Some(end),
            step: 1,
        }
    }

    /// Creates a strided slice `[start..end)` with explicit `step`.
    pub const fn stepped(start: usize, end: usize, step: usize) -> Self {
        Self {
            start,
            end: Some(end),
            step,
        }
    }
}

/// Complete memory layout of a tensor view: shape, element strides, and start offset.
///
/// Strides of `0` represent broadcasted dimensions (zero-copy expansion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    /// Constructs a contiguous row-major layout starting at offset 0.
    pub fn contiguous(shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Constructs a layout with explicit shape, strides, and offset.
    pub fn from_parts(shape: Shape, strides: Vec<usize>, offset: usize) -> Result<Self> {
        if shape.rank() != strides.len() {
            return Err(Error::InvalidRank {
                context: "Layout::from_parts",
                rank: strides.len(),
                expected: shape.rank(),
            });
        }
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    /// Returns the tensor's [`Shape`].
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the element strides per dimension.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the starting element offset into the underlying storage buffer.
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the total number of logical elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Returns `true` if elements are laid out in standard C-contiguous (row-major) order
    /// with no broadcasted (zero-stride) dimensions of size > 1.
    pub fn is_contiguous(&self) -> bool {
        if self.numel() <= 1 {
            return true;
        }
        let mut expected_stride = 1usize;
        for (&dim, &stride) in self.shape.dims().iter().rev().zip(self.strides.iter().rev()) {
            if dim == 1 {
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride = expected_stride.saturating_mul(dim);
        }
        true
    }

    /// Computes the physical storage index for a given flat logical index `0..self.numel()`.
    ///
    /// Uses checked arithmetic to prevent overflow on adversarial strides.
    pub fn physical_index(&self, mut logical_index: usize) -> Result<usize> {
        let dims = self.shape.dims();
        let mut phys = self.offset;
        for i in (0..dims.len()).rev() {
            let dim = dims.get(i).copied().unwrap_or(1);
            let stride = self.strides.get(i).copied().unwrap_or(0);
            if dim == 0 {
                return Err(Error::IndexOutOfBounds {
                    index: logical_index,
                    len: 0,
                });
            }
            let coord = logical_index % dim;
            logical_index /= dim;
            let delta = coord
                .checked_mul(stride)
                .ok_or(Error::ArithmeticOverflow {
                    context: "Layout::physical_index stride multiply",
                })?;
            phys = phys
                .checked_add(delta)
                .ok_or(Error::ArithmeticOverflow {
                    context: "Layout::physical_index offset add",
                })?;
        }
        Ok(phys)
    }

    /// Computes the physical storage index for explicit multi-dimensional coordinates.
    pub fn index_of(&self, coords: &[usize]) -> Result<usize> {
        if coords.len() != self.rank() {
            return Err(Error::InvalidRank {
                context: "Layout::index_of",
                rank: coords.len(),
                expected: self.rank(),
            });
        }
        let mut phys = self.offset;
        for ((&c, &dim), &stride) in coords
            .iter()
            .zip(self.shape.dims().iter())
            .zip(self.strides.iter())
        {
            if c >= dim {
                return Err(Error::IndexOutOfBounds { index: c, len: dim });
            }
            let delta = c.checked_mul(stride).ok_or(Error::ArithmeticOverflow {
                context: "Layout::index_of stride multiply",
            })?;
            phys = phys
                .checked_add(delta)
                .ok_or(Error::ArithmeticOverflow {
                    context: "Layout::index_of offset add",
                })?;
        }
        Ok(phys)
    }

    /// Returns a zero-copy broadcasted layout targeting `target_shape`.
    pub fn broadcast_to(&self, target_shape: &Shape) -> Result<Self> {
        let src_rank = self.rank();
        let dst_rank = target_shape.rank();
        if src_rank > dst_rank {
            return Err(Error::BroadcastIncompatible {
                lhs: self.shape.dims().to_vec(),
                rhs: target_shape.dims().to_vec(),
            });
        }
        let mut new_strides = vec![0usize; dst_rank];
        let rank_diff = dst_rank - src_rank;
        for i in 0..dst_rank {
            let dst_dim = target_shape.dims().get(i).copied().unwrap_or(1);
            if i < rank_diff {
                if let Some(slot) = new_strides.get_mut(i) {
                    *slot = 0;
                }
            } else {
                let src_idx = i - rank_diff;
                let src_dim = self.shape.dims().get(src_idx).copied().unwrap_or(1);
                let src_stride = self.strides.get(src_idx).copied().unwrap_or(0);
                if src_dim == dst_dim {
                    if let Some(slot) = new_strides.get_mut(i) {
                        *slot = src_stride;
                    }
                } else if src_dim == 1 {
                    if let Some(slot) = new_strides.get_mut(i) {
                        *slot = 0;
                    }
                } else {
                    return Err(Error::BroadcastIncompatible {
                        lhs: self.shape.dims().to_vec(),
                        rhs: target_shape.dims().to_vec(),
                    });
                }
            }
        }
        Ok(Self {
            shape: target_shape.clone(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Returns a zero-copy transposed layout swapping `axis_a` and `axis_b`.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let rank = self.rank();
        if axis_a >= rank {
            return Err(Error::InvalidAxis { axis: axis_a, rank });
        }
        if axis_b >= rank {
            return Err(Error::InvalidAxis { axis: axis_b, rank });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        dims.swap(axis_a, axis_b);
        strides.swap(axis_a, axis_b);
        Ok(Self {
            shape: Shape::new(dims)?,
            strides,
            offset: self.offset,
        })
    }

    /// Returns a zero-copy permuted layout according to `perm`.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let rank = self.rank();
        if perm.len() != rank {
            return Err(Error::InvalidPermutation {
                perm: perm.to_vec(),
                rank,
            });
        }
        let mut seen = vec![false; rank];
        let mut dims = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);
        for &axis in perm {
            let Some(flag) = seen.get_mut(axis) else {
                return Err(Error::InvalidPermutation {
                    perm: perm.to_vec(),
                    rank,
                });
            };
            if *flag {
                return Err(Error::InvalidPermutation {
                    perm: perm.to_vec(),
                    rank,
                });
            }
            *flag = true;
            dims.push(self.shape.dim(axis)?);
            strides.push(self.strides.get(axis).copied().unwrap_or(0));
        }
        Ok(Self {
            shape: Shape::new(dims)?,
            strides,
            offset: self.offset,
        })
    }

    /// Returns a zero-copy reshaped layout if `self` is contiguous and `new_shape`
    /// has the same total element count.
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        if self.numel() != new_shape.numel() {
            return Err(Error::ShapeMismatch {
                context: "Layout::reshape",
                expected: self.shape.dims().to_vec(),
                actual: new_shape.dims().to_vec(),
            });
        }
        if !self.is_contiguous() {
            return Err(Error::NonContiguousLayout {
                context: "Layout::reshape",
            });
        }
        let strides = new_shape.contiguous_strides();
        Ok(Self {
            shape: new_shape,
            strides,
            offset: self.offset,
        })
    }

    /// Inserts a size-1 dimension at `axis` (zero-copy).
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        let rank = self.rank();
        if axis > rank {
            return Err(Error::InvalidAxis {
                axis,
                rank: rank + 1,
            });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        let new_stride = if axis < rank {
            let d = self.shape.dim(axis)?;
            let s = self.strides.get(axis).copied().unwrap_or(1);
            d.saturating_mul(s)
        } else {
            1
        };
        dims.insert(axis, 1);
        strides.insert(axis, new_stride);
        Ok(Self {
            shape: Shape::new(dims)?,
            strides,
            offset: self.offset,
        })
    }

    /// Removes a size-1 dimension at `axis` (zero-copy).
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        let dim = self.shape.dim(axis)?;
        if dim != 1 {
            return Err(Error::InvalidArgument {
                context: "Layout::squeeze",
                reason: format!("cannot squeeze axis {axis} with dimension {dim} != 1"),
            });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        dims.remove(axis);
        strides.remove(axis);
        Ok(Self {
            shape: Shape::new(dims)?,
            strides,
            offset: self.offset,
        })
    }

    /// Returns a zero-copy sliced view according to per-axis [`SliceRange`] specifications.
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<Self> {
        if ranges.len() > self.rank() {
            return Err(Error::InvalidRank {
                context: "Layout::slice",
                rank: ranges.len(),
                expected: self.rank(),
            });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        let mut offset = self.offset;

        for (axis, &r) in ranges.iter().enumerate() {
            let dim = self.shape.dim(axis)?;
            let end = r.end.unwrap_or(dim);
            if r.step == 0 || r.start > end || end > dim {
                return Err(Error::InvalidSlice {
                    axis,
                    dim,
                    start: r.start,
                    end,
                    step: r.step,
                });
            }
            let len = end - r.start;
            let new_dim = if len == 0 {
                0
            } else {
                len.div_ceil(r.step)
            };
            let old_stride = strides.get(axis).copied().unwrap_or(0);
            let offset_delta = r
                .start
                .checked_mul(old_stride)
                .ok_or(Error::ArithmeticOverflow {
                    context: "Layout::slice offset",
                })?;
            offset = offset
                .checked_add(offset_delta)
                .ok_or(Error::ArithmeticOverflow {
                    context: "Layout::slice offset",
                })?;
            let new_stride = old_stride
                .checked_mul(r.step)
                .ok_or(Error::ArithmeticOverflow {
                    context: "Layout::slice stride",
                })?;
            if let Some(d_slot) = dims.get_mut(axis) {
                *d_slot = new_dim;
            }
            if let Some(s_slot) = strides.get_mut(axis) {
                *s_slot = new_stride;
            }
        }

        Ok(Self {
            shape: Shape::new(dims)?,
            strides,
            offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_overflow_and_rank_limits() {
        let err = Shape::new(vec![usize::MAX, 2]).unwrap_err();
        assert_eq!(
            err,
            Error::ArithmeticOverflow {
                context: "Shape::numel",
            }
        );

        let err = Shape::new(vec![1; MAX_RANK + 1]).unwrap_err();
        assert!(matches!(err, Error::InvalidRank { .. }));
    }

    #[test]
    fn broadcasting_rules() {
        let a = Shape::new(vec![2, 1, 4]).unwrap();
        let b = Shape::new(vec![3, 4]).unwrap();
        let c = a.broadcast_shape(&b).unwrap();
        assert_eq!(c.dims(), &[2, 3, 4]);

        let bad_a = Shape::new(vec![2, 3]).unwrap();
        let bad_b = Shape::new(vec![2, 4]).unwrap();
        assert!(matches!(
            bad_a.broadcast_shape(&bad_b),
            Err(Error::BroadcastIncompatible { .. })
        ));
    }

    #[test]
    fn transpose_and_slice_layouts() {
        let layout = Layout::contiguous(Shape::new(vec![3, 4]).unwrap());
        assert!(layout.is_contiguous());
        assert_eq!(layout.strides(), &[4, 1]);

        let t = layout.transpose(0, 1).unwrap();
        assert!(!t.is_contiguous());
        assert_eq!(t.shape().dims(), &[4, 3]);
        assert_eq!(t.strides(), &[1, 4]);
        // Element (2, 1) of transposed equals (1, 2) of original = 1*4 + 2 = 6
        assert_eq!(t.index_of(&[2, 1]).unwrap(), 6);

        let s = layout
            .slice(&[SliceRange::range(1, 3), SliceRange::stepped(0, 4, 2)])
            .unwrap();
        assert_eq!(s.shape().dims(), &[2, 2]);
        assert_eq!(s.offset(), 4);
        assert_eq!(s.strides(), &[4, 2]);
        assert_eq!(s.index_of(&[1, 1]).unwrap(), 10);
    }
}
