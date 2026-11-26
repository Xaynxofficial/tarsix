//! Static graph-level memory planner using interval-graph buffer slot reuse.
//!
//! Given the topological execution lifetime `[first_step, last_step]` and byte size of every
//! tensor in a compiled graph, [`plan_memory_reuse`] assigns non-overlapping tensors to shared
//! physical buffer slots using best-fit interval coloring.

use std::collections::BTreeMap;

use as_core::{Error, Result};

/// Execution lifetime interval for a single tensor in a compiled graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLifetime {
    /// Logical tensor / node identifier.
    pub id: usize,
    /// Required allocation size in bytes.
    pub byte_size: usize,
    /// First execution step at which this tensor is produced.
    pub first_step: usize,
    /// Last execution step at which this tensor is consumed (inclusive).
    pub last_step: usize,
    /// If `true`, this tensor is an external input or graph output and must not share a slot.
    pub pinned: bool,
}

/// Physical reusable memory slot assigned to one or more non-overlapping tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSlot {
    /// Unique slot identifier.
    pub slot_id: usize,
    /// Maximum byte size required across all tensors assigned to this slot.
    pub capacity_bytes: usize,
    /// Logical tensor IDs sharing this physical buffer slot.
    pub assigned_tensors: Vec<usize>,
}

/// Result of static graph memory planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    /// Physical reusable buffer slots for intermediate tensors.
    pub slots: Vec<BufferSlot>,
    /// Mapping from logical tensor ID -> assigned physical slot ID (non-pinned tensors).
    pub tensor_to_slot: BTreeMap<usize, usize>,
    /// Sum of byte sizes of all non-pinned intermediate tensors (no reuse at all).
    pub naive_total_bytes: usize,
    /// Peak concurrent live intermediate bytes across execution steps.
    pub naive_peak_bytes: usize,
    /// Total physical bytes required by the planned reusable slots.
    pub planned_peak_bytes: usize,
}

impl MemoryPlan {
    /// Fraction of total naive intermediate allocation bytes saved by slot reuse (`0.0..1.0`).
    pub fn memory_savings_ratio(&self) -> f64 {
        if self.naive_total_bytes == 0 {
            0.0
        } else {
            let saved = self
                .naive_total_bytes
                .saturating_sub(self.planned_peak_bytes);
            (saved as f64) / (self.naive_total_bytes as f64)
        }
    }

    /// Ratio of `naive_total_bytes / planned_peak_bytes` (`>= 1.0` when non-empty).
    pub fn reuse_factor(&self) -> f64 {
        if self.planned_peak_bytes == 0 {
            1.0
        } else {
            (self.naive_total_bytes as f64) / (self.planned_peak_bytes as f64)
        }
    }
}

/// Computes a static buffer-reuse [`MemoryPlan`] for `lifetimes` via best-fit interval coloring.
///
/// Guarantees that no two tensors assigned to the same [`BufferSlot`] have overlapping
/// `[first_step, last_step]` intervals.
pub fn plan_memory_reuse(lifetimes: &[TensorLifetime]) -> Result<MemoryPlan> {
    let mut intermediates = Vec::new();
    let mut naive_total_bytes = 0usize;
    let mut max_step = 0usize;

    for &lt in lifetimes {
        if lt.first_step > lt.last_step {
            return Err(Error::InvalidArgument {
                context: "plan_memory_reuse",
                reason: format!(
                    "tensor {} has inverted lifetime [{}, {}]",
                    lt.id, lt.first_step, lt.last_step
                ),
            });
        }
        if !lt.pinned && lt.byte_size > 0 {
            naive_total_bytes = naive_total_bytes
                .checked_add(lt.byte_size)
                .ok_or(Error::ArithmeticOverflow {
                    context: "plan_memory_reuse naive_total_bytes",
                })?;
            max_step = max_step.max(lt.last_step);
            intermediates.push(lt);
        }
    }

    if intermediates.is_empty() {
        return Ok(MemoryPlan {
            slots: Vec::new(),
            tensor_to_slot: BTreeMap::new(),
            naive_total_bytes: 0,
            naive_peak_bytes: 0,
            planned_peak_bytes: 0,
        });
    }

    // Compute exact concurrent peak across steps using sweep-line events
    let mut events: Vec<(usize, i8, usize)> = Vec::with_capacity(intermediates.len() * 2);
    for lt in &intermediates {
        // Start events (0) before end events (1) at the same step so inclusive endpoints count as live
        events.push((lt.first_step, 0, lt.byte_size));
        events.push((lt.last_step, 1, lt.byte_size));
    }
    events.sort_by_key(|&(step, kind, _)| (step, kind));

    let mut current_live = 0usize;
    let mut naive_peak_bytes = 0usize;
    for (_, kind, bytes) in events {
        if kind == 0 {
            current_live = current_live.saturating_add(bytes);
            naive_peak_bytes = naive_peak_bytes.max(current_live);
        } else {
            current_live = current_live.saturating_sub(bytes);
        }
    }

    // Sort intermediates by byte_size descending, then first_step ascending (largest-first best-fit)
    intermediates.sort_by(|a, b| {
        b.byte_size
            .cmp(&a.byte_size)
            .then_with(|| a.first_step.cmp(&b.first_step))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut slots: Vec<BufferSlot> = Vec::new();
    let mut slot_intervals: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut tensor_to_slot: BTreeMap<usize, usize> = BTreeMap::new();

    for lt in &intermediates {
        // Find non-overlapping slot with smallest capacity >= lt.byte_size (or minimal growth)
        let mut best_slot: Option<(usize, usize)> = None;
        for (idx, slot) in slots.iter().enumerate() {
            let intervals = slot_intervals.get(idx).map_or(&[][..], Vec::as_slice);
            let overlaps = intervals
                .iter()
                .any(|&(s, e)| lt.first_step <= e && s <= lt.last_step);
            if overlaps {
                continue;
            }
            let waste = slot.capacity_bytes.abs_diff(lt.byte_size);
            match best_slot {
                None => best_slot = Some((idx, waste)),
                Some((_, best_waste)) if waste < best_waste => best_slot = Some((idx, waste)),
                _ => {}
            }
        }

        if let Some((slot_idx, _)) = best_slot {
            if let Some(slot) = slots.get_mut(slot_idx) {
                slot.capacity_bytes = slot.capacity_bytes.max(lt.byte_size);
                slot.assigned_tensors.push(lt.id);
            }
            if let Some(intervals) = slot_intervals.get_mut(slot_idx) {
                intervals.push((lt.first_step, lt.last_step));
            }
            tensor_to_slot.insert(lt.id, slot_idx);
        } else {
            let slot_id = slots.len();
            slots.push(BufferSlot {
                slot_id,
                capacity_bytes: lt.byte_size,
                assigned_tensors: vec![lt.id],
            });
            slot_intervals.push(vec![(lt.first_step, lt.last_step)]);
            tensor_to_slot.insert(lt.id, slot_id);
        }
    }

    let mut planned_peak_bytes = 0usize;
    for slot in &slots {
        planned_peak_bytes = planned_peak_bytes
            .checked_add(slot.capacity_bytes)
            .ok_or(Error::ArithmeticOverflow {
                context: "plan_memory_reuse planned_peak_bytes",
            })?;
    }

    Ok(MemoryPlan {
        slots,
        tensor_to_slot,
        naive_total_bytes,
        naive_peak_bytes,
        planned_peak_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_slots_for_sequential_chain() {
        // Sequential 5-layer chain: each intermediate lives for 2 steps [i, i+1]
        // Only 2 concurrent buffers are ever live -> 2 slots instead of 5!
        let lifetimes = vec![
            TensorLifetime { id: 0, byte_size: 4096, first_step: 0, last_step: 1, pinned: true },
            TensorLifetime { id: 1, byte_size: 4096, first_step: 1, last_step: 2, pinned: false },
            TensorLifetime { id: 2, byte_size: 4096, first_step: 2, last_step: 3, pinned: false },
            TensorLifetime { id: 3, byte_size: 4096, first_step: 3, last_step: 4, pinned: false },
            TensorLifetime { id: 4, byte_size: 4096, first_step: 4, last_step: 5, pinned: false },
            TensorLifetime { id: 5, byte_size: 4096, first_step: 5, last_step: 6, pinned: false },
            TensorLifetime { id: 6, byte_size: 4096, first_step: 6, last_step: 6, pinned: true },
        ];

        let plan = plan_memory_reuse(&lifetimes).unwrap();
        assert_eq!(plan.naive_total_bytes, 5 * 4096);
        assert_eq!(plan.naive_peak_bytes, 2 * 4096);
        assert_eq!(plan.slots.len(), 2);
        assert_eq!(plan.planned_peak_bytes, 2 * 4096);
        assert!((plan.memory_savings_ratio() - 0.6).abs() < 1e-9);
        assert!((plan.reuse_factor() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn rejects_inverted_lifetime() {
        let bad = vec![TensorLifetime {
            id: 1,
            byte_size: 128,
            first_step: 5,
            last_step: 2,
            pinned: false,
        }];
        assert!(plan_memory_reuse(&bad).is_err());
    }
}
