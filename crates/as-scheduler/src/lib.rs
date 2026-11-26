//! Hardware Scheduler for CPU/GPU task management and parallel DAG execution (`as-scheduler`).
//!
//! Provides:
//! - [`DeviceRegistry`] and [`DeviceProfile`] for dynamic CPU and Apple Silicon Metal GPU discovery
//! - [`plan_placement`], [`PlacementPolicy`], and [`PlacementPlan`] for cost-model operator placement
//! - [`build_wave_schedule`] and [`WaveSchedule`] for parallel DAG wave partitioning
//! - [`HardwareScheduler`] for multi-device, wave-parallel graph execution

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod executor;
pub mod placement;
pub mod registry;
pub mod wave;

pub use executor::{HardwareScheduler, ScheduleExecutionReport};
pub use placement::{OpPlacement, PlacementPlan, PlacementPolicy, plan_placement};
pub use registry::{DeviceProfile, DeviceRegistry};
pub use wave::{ExecutionWave, WaveSchedule, build_wave_schedule};
