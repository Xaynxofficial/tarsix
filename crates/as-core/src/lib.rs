//! Core foundational types for the independent AI computing infrastructure.
//!
//! This crate defines:
//! - Strongly typed error handling ([`Error`], [`Result`])
//! - Resource and security limits ([`MemoryLimits`])
//! - Hardware device abstractions ([`Device`], [`DeviceCapabilities`])
//! - Numeric data types ([`DType`], [`F16`], [`BF16`], [`TensorElement`])
//! - N-dimensional [`Shape`], [`Layout`], broadcasting, and zero-copy view algebra

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod device;
pub mod dtype;
pub mod error;
pub mod limits;
pub mod shape;

pub use device::{Device, DeviceCapabilities};
pub use dtype::{BF16, DType, F16, TensorElement};
pub use error::{Error, Result};
pub use limits::{MAX_RANK, MemoryLimits};
pub use shape::{Layout, Shape, SliceRange};
