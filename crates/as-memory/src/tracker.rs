//! Thread-safe global/scoped memory tracking and OOM budget enforcement.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use as_core::{Error, Result};

/// Default global memory budget (8 GiB on 64-bit targets, 512 MiB on 32-bit/WASM).
#[cfg(target_pointer_width = "64")]
pub const DEFAULT_GLOBAL_MEMORY_BUDGET: usize = 8 * 1024 * 1024 * 1024;

/// Default global memory budget on 32-bit targets (512 MiB).
#[cfg(not(target_pointer_width = "64"))]
pub const DEFAULT_GLOBAL_MEMORY_BUDGET: usize = 512 * 1024 * 1024;

/// Configurable global memory ceiling for a runtime or session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Maximum total live bytes permitted concurrently across tracked allocations.
    pub max_total_bytes: usize,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_total_bytes: DEFAULT_GLOBAL_MEMORY_BUDGET,
        }
    }
}

impl MemoryBudget {
    /// Creates a new [`MemoryBudget`] with an explicit byte limit.
    pub const fn new(max_total_bytes: usize) -> Self {
        Self { max_total_bytes }
    }
}

/// Point-in-time snapshot of [`MemoryTracker`] telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Currently allocated live bytes.
    pub live_bytes: usize,
    /// High-water mark of concurrent live bytes.
    pub peak_bytes: usize,
    /// Cumulative bytes allocated since tracker creation.
    pub total_allocated_bytes: u64,
    /// Cumulative bytes released since tracker creation.
    pub total_freed_bytes: u64,
    /// Total number of allocation reservations succeeded.
    pub allocation_count: u64,
    /// Total number of releases recorded.
    pub deallocation_count: u64,
    /// Number of buffer pool cache hits.
    pub pool_hits: u64,
    /// Number of buffer pool cache misses.
    pub pool_misses: u64,
    /// Configured global byte budget.
    pub budget_bytes: usize,
}

impl MemorySnapshot {
    /// Returns fraction of the configured budget currently occupied (`0.0..=1.0`).
    pub fn utilization_ratio(&self) -> f64 {
        if self.budget_bytes == 0 {
            0.0
        } else {
            (self.live_bytes as f64) / (self.budget_bytes as f64)
        }
    }
}

/// Lock-free, thread-safe memory telemetry and budget enforcer.
#[derive(Debug)]
pub struct MemoryTracker {
    budget: MemoryBudget,
    live_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    total_allocated_bytes: AtomicU64,
    total_freed_bytes: AtomicU64,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new(MemoryBudget::default())
    }
}

impl MemoryTracker {
    /// Creates a new [`MemoryTracker`] enforcing `budget`.
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            budget,
            live_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            total_allocated_bytes: AtomicU64::new(0),
            total_freed_bytes: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            pool_misses: AtomicU64::new(0),
        }
    }

    /// Returns the configured [`MemoryBudget`].
    pub const fn budget(&self) -> MemoryBudget {
        self.budget
    }

    /// Attempts to reserve `bytes` from the global budget using a lock-free CAS loop.
    ///
    /// Returns [`Error::AllocationLimitExceeded`] if adding `bytes` would exceed the budget.
    pub fn try_reserve(&self, bytes: usize) -> Result<()> {
        let mut current = self.live_bytes.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .ok_or(Error::ArithmeticOverflow {
                    context: "MemoryTracker::try_reserve",
                })?;
            if next > self.budget.max_total_bytes {
                return Err(Error::AllocationLimitExceeded {
                    requested_bytes: next,
                    limit_bytes: self.budget.max_total_bytes,
                });
            }
            match self.live_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.update_peak(next);
                    self.total_allocated_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    self.allocation_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                }
                Err(actual) => current = actual,
            }
        }
    }

    /// Releases `bytes` previously reserved from the tracker (saturating at 0).
    pub fn release(&self, bytes: usize) {
        let mut current = self.live_bytes.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(bytes);
            match self.live_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.total_freed_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    self.deallocation_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(actual) => current = actual,
            }
        }
    }

    /// Reserves `bytes` and returns an RAII [`MemoryLease`] that releases on drop.
    pub fn lease(self: &Arc<Self>, bytes: usize) -> Result<MemoryLease> {
        self.try_reserve(bytes)?;
        Ok(MemoryLease {
            tracker: Arc::clone(self),
            bytes,
        })
    }

    /// Records a buffer pool cache hit.
    pub fn record_pool_hit(&self) {
        self.pool_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a buffer pool cache miss.
    pub fn record_pool_miss(&self) {
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Resets the recorded peak memory to the current live byte count.
    pub fn reset_peak(&self) {
        let live = self.live_bytes.load(Ordering::Acquire);
        self.peak_bytes.store(live, Ordering::Release);
    }

    /// Captures a [`MemorySnapshot`] of current telemetry.
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            live_bytes: self.live_bytes.load(Ordering::Acquire),
            peak_bytes: self.peak_bytes.load(Ordering::Acquire),
            total_allocated_bytes: self.total_allocated_bytes.load(Ordering::Relaxed),
            total_freed_bytes: self.total_freed_bytes.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            pool_misses: self.pool_misses.load(Ordering::Relaxed),
            budget_bytes: self.budget.max_total_bytes,
        }
    }

    fn update_peak(&self, candidate: usize) {
        let mut peak = self.peak_bytes.load(Ordering::Acquire);
        while candidate > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                candidate,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }
}

/// RAII lease holding a memory reservation on a [`MemoryTracker`].
#[derive(Debug)]
pub struct MemoryLease {
    tracker: Arc<MemoryTracker>,
    bytes: usize,
}

impl MemoryLease {
    /// Returns the number of bytes reserved by this lease.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        self.tracker.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforces_budget_and_tracks_peak() {
        let tracker = Arc::new(MemoryTracker::new(MemoryBudget::new(1024)));
        {
            let lease1 = tracker.lease(600).unwrap();
            assert_eq!(lease1.bytes(), 600);
            let err = tracker.lease(500).unwrap_err();
            assert_eq!(
                err,
                Error::AllocationLimitExceeded {
                    requested_bytes: 1100,
                    limit_bytes: 1024,
                }
            );
            let lease2 = tracker.lease(400).unwrap();
            assert_eq!(tracker.snapshot().live_bytes, 1000);
            assert_eq!(tracker.snapshot().peak_bytes, 1000);
            drop(lease2);
            assert_eq!(tracker.snapshot().live_bytes, 600);
            assert_eq!(tracker.snapshot().peak_bytes, 1000);
        }
        let snap = tracker.snapshot();
        assert_eq!(snap.live_bytes, 0);
        assert_eq!(snap.peak_bytes, 1000);
        assert_eq!(snap.allocation_count, 2);
        assert_eq!(snap.deallocation_count, 2);
    }
}
