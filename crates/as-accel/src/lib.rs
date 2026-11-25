//! Hardware acceleration architecture:
//! - Host and accelerator capability discovery ([`discover_backends`], [`BackendDescriptor`])
//! - Portable GPU compute kernel sources for Metal (MSL), WebGPU (WGSL), and CUDA C++ ([`kernels`])
//! - Cross-backend numerical consistency test harness ([`verify_backend_consistency`])

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod consistency;
pub mod kernels;
pub mod registry;

pub use consistency::{ConsistencyReport, verify_backend_consistency};
pub use registry::{BackendDescriptor, BackendKind, discover_backends};
