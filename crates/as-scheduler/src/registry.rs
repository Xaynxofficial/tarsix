//! Dynamic hardware device discovery and capability registry.

use as_core::Device;
use as_metal::MetalBackend;

/// Hardware performance and availability profile for a logical [`Device`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceProfile {
    /// Logical device identifier.
    pub device: Device,
    /// Human-readable device name.
    pub name: String,
    /// Whether this device can execute operations on this host right now.
    pub available: bool,
    /// Logical compute units (CPU threads or estimated GPU compute cores).
    pub compute_units: usize,
    /// Estimated peak single-precision throughput in GFLOP/s (used by cost model).
    pub peak_gflops_estimate: f64,
    /// Estimated memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// Fixed launch/synchronization overhead per dispatch in microseconds.
    pub launch_overhead_us: f64,
    /// Host↔Device transfer bandwidth in GB/s (`f64::INFINITY` for CPU / unified memory).
    pub transfer_bandwidth_gbs: f64,
}

/// Registry of discovered and user-configured hardware execution devices.
#[derive(Debug, Clone)]
pub struct DeviceRegistry {
    devices: Vec<DeviceProfile>,
}

impl DeviceRegistry {
    /// Probes the host machine and builds a [`DeviceRegistry`] with live CPU and Metal GPU status.
    pub fn probe_host() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);

        let mut devices = vec![DeviceProfile {
            device: Device::Cpu,
            name: format!(
                "CPU ({} / {}, {} threads)",
                std::env::consts::ARCH,
                std::env::consts::OS,
                cpu_threads
            ),
            available: true,
            compute_units: cpu_threads,
            peak_gflops_estimate: (cpu_threads as f64) * 12.0,
            memory_bandwidth_gbs: 60.0,
            launch_overhead_us: 1.0,
            transfer_bandwidth_gbs: f64::INFINITY,
        }];

        if let Ok(metal) = MetalBackend::new() {
            let dev_name = metal.device_name().to_owned();
            devices.push(DeviceProfile {
                device: Device::Metal(0),
                name: format!("Apple Metal GPU ({dev_name})"),
                available: true,
                compute_units: 16,
                peak_gflops_estimate: 2_500.0,
                memory_bandwidth_gbs: 200.0,
                launch_overhead_us: 45.0,
                transfer_bandwidth_gbs: 120.0,
            });
        }

        Self { devices }
    }

    /// Creates a registry from an explicit list of [`DeviceProfile`]s (useful for testing or simulation).
    pub fn from_profiles(devices: Vec<DeviceProfile>) -> Self {
        Self { devices }
    }

    /// Returns all registered device profiles.
    pub fn devices(&self) -> &[DeviceProfile] {
        &self.devices
    }

    /// Looks up the profile for a specific [`Device`].
    pub fn get(&self, device: Device) -> Option<&DeviceProfile> {
        self.devices.iter().find(|d| d.device == device)
    }

    /// Returns the fastest available non-CPU accelerator, if any.
    pub fn best_accelerator(&self) -> Option<&DeviceProfile> {
        self.devices
            .iter()
            .filter(|d| d.available && d.device != Device::Cpu)
            .max_by(|a, b| {
                a.peak_gflops_estimate
                    .total_cmp(&b.peak_gflops_estimate)
            })
    }
}
