//! Independent, zero-dependency, memory-safe Memory Engine (`as-memory`).
//!
//! Provides:
//! - [`MemoryTracker`], [`MemoryBudget`], [`MemoryLease`], and [`MemorySnapshot`] for
//!   thread-safe live/peak byte telemetry and global OOM budget enforcement
//! - [`BufferPool`] and [`TensorBufferPool`] for size-class buffer recycling with bounded cache ceilings
//! - [`plan_memory_reuse`], [`MemoryPlan`], and [`TensorLifetime`] for static graph-level
//!   buffer slot reuse via interval coloring

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod planner;
pub mod pool;
pub mod tracker;

pub use planner::{BufferSlot, MemoryPlan, TensorLifetime, plan_memory_reuse};
pub use pool::{BufferPool, PoolConfig, PoolStats, TensorBufferPool};
pub use tracker::{MemoryBudget, MemoryLease, MemorySnapshot, MemoryTracker};
