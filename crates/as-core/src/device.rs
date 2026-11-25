//! Hardware device abstraction and capability descriptors.

use core::fmt;

/// Logical execution device identifier.
///
/// Milestone 1 implements [`Device::Cpu`]. Additional variants are reserved
/// so upper layers remain hardware-independent across future phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host CPU backend.
    #[default]
    Cpu,
    /// Apple Silicon Metal GPU backend (Phase 7).
    Metal(usize),
    /// NVIDIA CUDA GPU backend (Phase 7).
    Cuda(usize),
    /// WebGPU portable GPU backend (Phase 7).
    WebGpu(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Metal(idx) => write!(f, "metal:{idx}"),
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
            Self::WebGpu(idx) => write!(f, "webgpu:{idx}"),
        }
    }
}

/// Hardware capability metadata exposed by a backend implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    /// Target device identifier.
    pub device: Device,
    /// Human-readable backend name.
    pub name: String,
    /// Whether native FP16 arithmetic is supported in hardware.
    pub native_f16: bool,
    /// Whether native BF16 arithmetic is supported in hardware.
    pub native_bf16: bool,
    /// Whether deterministic execution mode is guaranteed.
    pub deterministic: bool,
}
