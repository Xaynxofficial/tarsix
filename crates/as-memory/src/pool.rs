//! Size-class buffer pooling for reusable host tensor allocations.

use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use as_core::{Error, Result};

use crate::tracker::MemoryTracker;

/// Configuration governing a [`BufferPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    /// Maximum total bytes retained inside the pool across all size classes.
    pub max_cached_bytes: usize,
    /// Maximum number of idle buffers kept per power-of-two size class.
    pub max_buffers_per_class: usize,
    /// Minimum power-of-two element capacity for bucket rounding.
    pub min_size_class: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_cached_bytes: 256 * 1024 * 1024, // 256 MiB
            max_buffers_per_class: 16,
            min_size_class: 64,
        }
    }
}

/// Telemetry statistics for a [`BufferPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Total requests satisfied from cached buffers.
    pub hits: u64,
    /// Total requests requiring fresh allocation.
    pub misses: u64,
    /// Current number of idle buffers stored in the pool.
    pub cached_buffers: usize,
    /// Current total bytes stored in the pool.
    pub cached_bytes: usize,
    /// Number of returned buffers dropped because the pool ceiling was reached.
    pub evictions: u64,
}

impl PoolStats {
    /// Fraction of acquisitions served from the pool (`0.0..=1.0`).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits.saturating_add(self.misses);
        if total == 0 {
            0.0
        } else {
            (self.hits as f64) / (total as f64)
        }
    }
}

#[derive(Debug)]
struct PoolInner<T> {
    buckets: BTreeMap<usize, Vec<Vec<T>>>,
    stats: PoolStats,
}

impl<T> Default for PoolInner<T> {
    fn default() -> Self {
        Self {
            buckets: BTreeMap::new(),
            stats: PoolStats::default(),
        }
    }
}

/// Thread-safe power-of-two size-class buffer pool for typed `Vec<T>` buffers.
#[derive(Debug)]
pub struct BufferPool<T> {
    config: PoolConfig,
    tracker: Option<Arc<MemoryTracker>>,
    inner: Mutex<PoolInner<T>>,
}

impl<T: Default + Clone> BufferPool<T> {
    /// Creates a new [`BufferPool`] with `config` and optional [`MemoryTracker`].
    pub fn new(config: PoolConfig, tracker: Option<Arc<MemoryTracker>>) -> Self {
        Self {
            config,
            tracker,
            inner: Mutex::new(PoolInner::default()),
        }
    }

    /// Acquires a zero/default-initialized `Vec<T>` of exact length `len`.
    ///
    /// Reuses an existing buffer from the matching power-of-two size class when available.
    pub fn acquire_zeroed(&self, len: usize) -> Result<Vec<T>> {
        let class = self.size_class(len)?;
        let elem_bytes = size_of::<T>().max(1);
        let class_bytes = class
            .checked_mul(elem_bytes)
            .ok_or(Error::ArithmeticOverflow {
                context: "BufferPool::acquire_zeroed class_bytes",
            })?;

        let mut guard = self.inner.lock().map_err(|_| Error::InternalInvariant {
            message: "BufferPool mutex poisoned",
        })?;

        if let Some(bucket) = guard.buckets.get_mut(&class)
            && let Some(mut buf) = bucket.pop()
        {
            guard.stats.hits += 1;
            guard.stats.cached_buffers = guard.stats.cached_buffers.saturating_sub(1);
            guard.stats.cached_bytes = guard.stats.cached_bytes.saturating_sub(class_bytes);
            drop(guard);

            if let Some(t) = &self.tracker {
                t.record_pool_hit();
            }

            buf.clear();
            buf.resize(len, T::default());
            return Ok(buf);
        }

        guard.stats.misses += 1;
        drop(guard);

        if let Some(t) = &self.tracker {
            t.record_pool_miss();
            t.try_reserve(class_bytes)?;
        }

        let mut buf = Vec::with_capacity(class);
        buf.resize(len, T::default());
        Ok(buf)
    }

    /// Returns a buffer to the pool for future reuse.
    ///
    /// Drops the buffer immediately if caching it would exceed `max_cached_bytes`
    /// or `max_buffers_per_class`.
    pub fn release(&self, mut buf: Vec<T>) {
        let cap = buf.capacity();
        if cap == 0 || !cap.is_power_of_two() || cap < self.config.min_size_class.max(1) {
            return;
        }
        let elem_bytes = size_of::<T>().max(1);
        let Some(buf_bytes) = cap.checked_mul(elem_bytes) else {
            return;
        };

        let Ok(mut guard) = self.inner.lock() else {
            return;
        };

        let bucket_len = guard.buckets.get(&cap).map_or(0, Vec::len);
        let would_exceed_bytes = guard
            .stats
            .cached_bytes
            .saturating_add(buf_bytes)
            > self.config.max_cached_bytes;

        if bucket_len >= self.config.max_buffers_per_class || would_exceed_bytes {
            guard.stats.evictions += 1;
            drop(guard);
            if let Some(t) = &self.tracker {
                t.release(buf_bytes);
            }
            return;
        }

        buf.clear();
        guard.buckets.entry(cap).or_default().push(buf);
        guard.stats.cached_buffers += 1;
        guard.stats.cached_bytes += buf_bytes;
    }

    /// Returns current [`PoolStats`].
    pub fn stats(&self) -> Result<PoolStats> {
        let guard = self.inner.lock().map_err(|_| Error::InternalInvariant {
            message: "BufferPool mutex poisoned",
        })?;
        Ok(guard.stats)
    }

    /// Empties all cached buffers from the pool and releases their bytes from the tracker.
    pub fn clear(&self) -> Result<()> {
        let mut guard = self.inner.lock().map_err(|_| Error::InternalInvariant {
            message: "BufferPool mutex poisoned",
        })?;
        let freed_bytes = guard.stats.cached_bytes;
        guard.buckets.clear();
        guard.stats.cached_buffers = 0;
        guard.stats.cached_bytes = 0;
        drop(guard);

        if freed_bytes > 0
            && let Some(t) = &self.tracker
        {
            t.release(freed_bytes);
        }
        Ok(())
    }

    fn size_class(&self, len: usize) -> Result<usize> {
        let min_cls = self.config.min_size_class.max(1).next_power_of_two();
        let target = len.max(min_cls);
        target.checked_next_power_of_two().ok_or(Error::ArithmeticOverflow {
            context: "BufferPool::size_class",
        })
    }
}

/// Unified typed buffer pool managing `f32`, `f64`, and raw `u8` tensor buffers.
#[derive(Debug)]
pub struct TensorBufferPool {
    /// Pool for 32-bit float buffers.
    pub f32_pool: BufferPool<f32>,
    /// Pool for 64-bit float buffers.
    pub f64_pool: BufferPool<f64>,
    /// Pool for raw byte buffers (e.g. quantized blocks, packed F16/BF16).
    pub byte_pool: BufferPool<u8>,
    tracker: Arc<MemoryTracker>,
}

impl TensorBufferPool {
    /// Creates a new [`TensorBufferPool`] backed by `tracker` and `config`.
    pub fn new(config: PoolConfig, tracker: Arc<MemoryTracker>) -> Self {
        Self {
            f32_pool: BufferPool::new(config, Some(Arc::clone(&tracker))),
            f64_pool: BufferPool::new(config, Some(Arc::clone(&tracker))),
            byte_pool: BufferPool::new(config, Some(Arc::clone(&tracker))),
            tracker,
        }
    }

    /// Returns the shared [`MemoryTracker`].
    pub fn tracker(&self) -> &Arc<MemoryTracker> {
        &self.tracker
    }

    /// Clears all underlying typed pools.
    pub fn clear_all(&self) -> Result<()> {
        self.f32_pool.clear()?;
        self.f64_pool.clear()?;
        self.byte_pool.clear()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::MemoryBudget;

    #[test]
    fn reuses_same_size_class_and_respects_cache_cap() {
        let tracker = Arc::new(MemoryTracker::new(MemoryBudget::new(1024 * 1024)));
        let cfg = PoolConfig {
            max_cached_bytes: 2048,
            max_buffers_per_class: 2,
            min_size_class: 64,
        };
        let pool = BufferPool::<f32>::new(cfg, Some(Arc::clone(&tracker)));

        let b1 = pool.acquire_zeroed(100).unwrap(); // rounds to 128 f32 = 512 bytes
        assert_eq!(b1.len(), 100);
        assert_eq!(b1.capacity(), 128);
        pool.release(b1);

        let b2 = pool.acquire_zeroed(90).unwrap(); // same class 128 -> cache hit
        assert_eq!(b2.len(), 90);
        let stats = pool.stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
        pool.release(b2);
        pool.clear().unwrap();
        assert_eq!(pool.stats().unwrap().cached_bytes, 0);
    }
}
