//! Hardware backend discovery and capability registry.

use as_core::Device;

/// Supported hardware execution backend families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Multi-core safe CPU backend.
    Cpu,
    /// Apple Silicon Metal GPU backend.
    Metal,
    /// Portable WebGPU compute backend (native & browser).
    WebGpu,
    /// NVIDIA CUDA GPU backend.
    Cuda,
}

/// Discovered hardware backend descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDescriptor {
    /// Target logical device.
    pub device: Device,
    /// Backend architecture family.
    pub kind: BackendKind,
    /// Human-readable hardware description.
    pub name: String,
    /// Whether this backend can execute immediately in the current build.
    pub ready_for_execution: bool,
    /// Logical parallel compute units (CPU threads or estimated GPU lanes).
    pub logical_units: usize,
}

/// Probes the host environment and reports available/supported hardware backends.
pub fn discover_backends() -> Vec<BackendDescriptor> {
    let cpu_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    let mut list = vec![BackendDescriptor {
        device: Device::Cpu,
        kind: BackendKind::Cpu,
        name: format!(
            "CPU ({} / {}, {} threads)",
            std::env::consts::ARCH,
            std::env::consts::OS,
            cpu_threads
        ),
        ready_for_execution: true,
        logical_units: cpu_threads,
    }];

    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        list.push(BackendDescriptor {
            device: Device::Metal(0),
            kind: BackendKind::Metal,
            name: "Apple Silicon Metal (MSL kernel ready)".to_owned(),
            ready_for_execution: false,
            logical_units: 0,
        });
    }

    list.push(BackendDescriptor {
        device: Device::WebGpu(0),
        kind: BackendKind::WebGpu,
        name: "WebGPU Portable Compute (WGSL shader ready)".to_owned(),
        ready_for_execution: false,
        logical_units: 0,
    });

    list.push(BackendDescriptor {
        device: Device::Cuda(0),
        kind: BackendKind::Cuda,
        name: "NVIDIA CUDA (NVRTC/PTX kernel ready)".to_owned(),
        ready_for_execution: false,
        logical_units: 0,
    });

    list
}
