//! Paged KV-Cache & PagedAttention engine (`PagedKvCache`).
//!
//! Inspired by vLLM's PagedAttention:
//! - Partitions KV-Cache memory into fixed-size physical blocks (`block_size` tokens per block)
//! - Eliminates contiguous pre-allocation fragmentation across concurrent sequences
//! - Supports zero-copy **Prefix Sharing + Copy-on-Write (CoW)** via block reference counting
//!   so multiple sequences sharing a system prompt share physical KV pages until divergence
//! - Computes exact scaled dot-product attention directly over non-contiguous physical pages

use std::collections::BTreeMap;

use as_core::{Error, Result};

/// Configuration for a [`PagedKvCache`] block pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvConfig {
    /// Number of Transformer decoder layers.
    pub num_layers: usize,
    /// Number of KV attention heads per layer.
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Number of tokens stored in a single physical KV block (e.g. 16).
    pub block_size: usize,
    /// Total number of physical blocks managed by the pool.
    pub num_blocks: usize,
}

/// Telemetry snapshot of a [`PagedKvCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedCacheStats {
    /// Total physical blocks in the pool.
    pub total_blocks: usize,
    /// Blocks currently allocated to at least one sequence.
    pub used_blocks: usize,
    /// Free blocks immediately available for allocation.
    pub free_blocks: usize,
    /// Physical blocks shared by 2 or more sequences (prefix sharing / CoW).
    pub shared_blocks: usize,
    /// Active sequences tracked by the page table.
    pub active_sequences: usize,
}

impl PagedCacheStats {
    /// Fraction of physical blocks currently in use (`0.0..=1.0`).
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            (self.used_blocks as f64) / (self.total_blocks as f64)
        }
    }
}

#[derive(Debug, Clone)]
struct PhysicalBlock {
    ref_count: usize,
    /// Layout: `[num_layers][block_size][num_kv_heads * head_dim]`
    k: Vec<f32>,
    /// Layout: `[num_layers][block_size][num_kv_heads * head_dim]`
    v: Vec<f32>,
}

#[derive(Debug, Clone)]
struct SequenceEntry {
    /// Ordered list of physical block IDs holding this sequence's tokens.
    block_table: Vec<usize>,
    /// Number of tokens currently populated across all layers (advances on layer `num_layers - 1`).
    seq_len: usize,
    /// Per-layer token counts so multi-layer appends advance consistently.
    layer_lens: Vec<usize>,
}

/// Paged Key-Value Cache with Copy-on-Write prefix sharing and direct PagedAttention.
#[derive(Debug)]
pub struct PagedKvCache {
    config: PagedKvConfig,
    kv_stride: usize,
    layer_stride: usize,
    blocks: Vec<PhysicalBlock>,
    free_list: Vec<usize>,
    sequences: BTreeMap<u64, SequenceEntry>,
}

impl PagedKvCache {
    /// Allocates a [`PagedKvCache`] block pool with `config`.
    pub fn new(config: PagedKvConfig) -> Result<Self> {
        if config.num_layers == 0
            || config.num_kv_heads == 0
            || config.head_dim == 0
            || config.block_size == 0
            || config.num_blocks == 0
        {
            return Err(Error::InvalidArgument {
                context: "PagedKvCache::new",
                reason: "all PagedKvConfig dimensions must be >= 1".to_owned(),
            });
        }
        let kv_stride = config
            .num_kv_heads
            .checked_mul(config.head_dim)
            .ok_or(Error::ArithmeticOverflow {
                context: "PagedKvCache kv_stride",
            })?;
        let layer_stride = config
            .block_size
            .checked_mul(kv_stride)
            .ok_or(Error::ArithmeticOverflow {
                context: "PagedKvCache layer_stride",
            })?;
        let block_elems = config
            .num_layers
            .checked_mul(layer_stride)
            .ok_or(Error::ArithmeticOverflow {
                context: "PagedKvCache block_elems",
            })?;

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            blocks.push(PhysicalBlock {
                ref_count: 0,
                k: vec![0.0f32; block_elems],
                v: vec![0.0f32; block_elems],
            });
        }
        let free_list: Vec<usize> = (0..config.num_blocks).rev().collect();

        Ok(Self {
            config,
            kv_stride,
            layer_stride,
            blocks,
            free_list,
            sequences: BTreeMap::new(),
        })
    }

    /// Returns the pool configuration.
    pub const fn config(&self) -> PagedKvConfig {
        self.config
    }

    /// Registers a new empty sequence `seq_id`.
    pub fn allocate_sequence(&mut self, seq_id: u64) -> Result<()> {
        if self.sequences.contains_key(&seq_id) {
            return Err(Error::InvalidArgument {
                context: "PagedKvCache::allocate_sequence",
                reason: format!("sequence {seq_id} already exists"),
            });
        }
        self.sequences.insert(
            seq_id,
            SequenceEntry {
                block_table: Vec::new(),
                seq_len: 0,
                layer_lens: vec![0; self.config.num_layers],
            },
        );
        Ok(())
    }

    /// Zero-copy forks `src_seq` into `dst_seq`, sharing all physical KV blocks via
    /// reference counting (Copy-on-Write prefix sharing).
    pub fn fork_sequence(&mut self, src_seq: u64, dst_seq: u64) -> Result<()> {
        if self.sequences.contains_key(&dst_seq) {
            return Err(Error::InvalidArgument {
                context: "PagedKvCache::fork_sequence",
                reason: format!("destination sequence {dst_seq} already exists"),
            });
        }
        let src = self
            .sequences
            .get(&src_seq)
            .cloned()
            .ok_or_else(|| Error::InvalidArgument {
                context: "PagedKvCache::fork_sequence",
                reason: format!("unknown source sequence {src_seq}"),
            })?;

        for &blk_id in &src.block_table {
            if let Some(blk) = self.blocks.get_mut(blk_id) {
                blk.ref_count += 1;
            }
        }
        self.sequences.insert(dst_seq, src);
        Ok(())
    }

    /// Frees `seq_id` and decrements reference counts on its physical blocks, returning
    /// any unreferenced blocks to the free pool.
    pub fn free_sequence(&mut self, seq_id: u64) -> Result<()> {
        let entry = self
            .sequences
            .remove(&seq_id)
            .ok_or_else(|| Error::InvalidArgument {
                context: "PagedKvCache::free_sequence",
                reason: format!("unknown sequence {seq_id}"),
            })?;
        for blk_id in entry.block_table {
            if let Some(blk) = self.blocks.get_mut(blk_id) {
                blk.ref_count = blk.ref_count.saturating_sub(1);
                if blk.ref_count == 0 {
                    self.free_list.push(blk_id);
                }
            }
        }
        Ok(())
    }

    /// Returns the number of cached tokens for `seq_id` at `layer`.
    pub fn seq_len(&self, seq_id: u64, layer: usize) -> Result<usize> {
        let entry = self
            .sequences
            .get(&seq_id)
            .ok_or_else(|| Error::InvalidArgument {
                context: "PagedKvCache::seq_len",
                reason: format!("unknown sequence {seq_id}"),
            })?;
        entry
            .layer_lens
            .get(layer)
            .copied()
            .ok_or(Error::IndexOutOfBounds {
                index: layer,
                len: self.config.num_layers,
            })
    }

    /// Appends one token's K and V vectors (each length `num_kv_heads * head_dim`) for `seq_id` at `layer`.
    ///
    /// Automatically allocates a new physical page when crossing a `block_size` boundary,
    /// and performs **Copy-on-Write** if the target page is currently shared with another sequence.
    pub fn append_token_kv(
        &mut self,
        seq_id: u64,
        layer: usize,
        k_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<()> {
        if layer >= self.config.num_layers {
            return Err(Error::IndexOutOfBounds {
                index: layer,
                len: self.config.num_layers,
            });
        }
        if k_vec.len() != self.kv_stride || v_vec.len() != self.kv_stride {
            return Err(Error::BufferLengthMismatch {
                expected: self.kv_stride,
                actual: k_vec.len(),
            });
        }

        let pos = self.seq_len(seq_id, layer)?;
        let logical_block = pos / self.config.block_size;
        let offset_in_block = pos % self.config.block_size;

        // Ensure physical block exists for this logical block
        let need_new_block = self
            .sequences
            .get(&seq_id)
            .is_none_or(|e| logical_block >= e.block_table.len());

        if need_new_block {
            let new_blk_id = self.alloc_block()?;
            if let Some(entry) = self.sequences.get_mut(&seq_id) {
                entry.block_table.push(new_blk_id);
            }
        }

        // Copy-on-Write: if the target physical block is shared (ref_count > 1), clone it first!
        let current_blk_id = self
            .sequences
            .get(&seq_id)
            .and_then(|e| e.block_table.get(logical_block).copied())
            .ok_or(Error::InternalInvariant {
                message: "PagedKvCache missing logical block",
            })?;

        let target_blk_id = if self
            .blocks
            .get(current_blk_id)
            .is_some_and(|b| b.ref_count > 1)
        {
            let cloned_id = self.alloc_block()?;
            let (src_k, src_v) = {
                let src = self.blocks.get(current_blk_id).ok_or(Error::InternalInvariant {
                    message: "CoW source block missing",
                })?;
                (src.k.clone(), src.v.clone())
            };
            if let Some(dst) = self.blocks.get_mut(cloned_id) {
                dst.k = src_k;
                dst.v = src_v;
            }
            if let Some(src) = self.blocks.get_mut(current_blk_id) {
                src.ref_count = src.ref_count.saturating_sub(1);
            }
            if let Some(entry) = self.sequences.get_mut(&seq_id)
                && let Some(slot) = entry.block_table.get_mut(logical_block)
            {
                *slot = cloned_id;
            }
            cloned_id
        } else {
            current_blk_id
        };

        let start = layer * self.layer_stride + offset_in_block * self.kv_stride;
        let end = start + self.kv_stride;
        let blk = self
            .blocks
            .get_mut(target_blk_id)
            .ok_or(Error::InternalInvariant {
                message: "target physical block missing",
            })?;
        if let Some(k_dst) = blk.k.get_mut(start..end) {
            k_dst.copy_from_slice(k_vec);
        }
        if let Some(v_dst) = blk.v.get_mut(start..end) {
            v_dst.copy_from_slice(v_vec);
        }

        if let Some(entry) = self.sequences.get_mut(&seq_id) {
            if let Some(l) = entry.layer_lens.get_mut(layer) {
                *l += 1;
            }
            entry.seq_len = entry.layer_lens.iter().copied().max().unwrap_or(0);
        }
        Ok(())
    }

    /// Computes exact **PagedAttention** for a single query head `q_head` (`len == head_dim`)
    /// attending over all cached tokens of `seq_id` at `(layer, kv_head)` across non-contiguous pages.
    pub fn paged_attention(
        &self,
        seq_id: u64,
        layer: usize,
        kv_head: usize,
        q_head: &[f32],
    ) -> Result<Vec<f32>> {
        if layer >= self.config.num_layers {
            return Err(Error::IndexOutOfBounds {
                index: layer,
                len: self.config.num_layers,
            });
        }
        if kv_head >= self.config.num_kv_heads {
            return Err(Error::IndexOutOfBounds {
                index: kv_head,
                len: self.config.num_kv_heads,
            });
        }
        if q_head.len() != self.config.head_dim {
            return Err(Error::BufferLengthMismatch {
                expected: self.config.head_dim,
                actual: q_head.len(),
            });
        }

        let entry = self
            .sequences
            .get(&seq_id)
            .ok_or_else(|| Error::InvalidArgument {
                context: "PagedKvCache::paged_attention",
                reason: format!("unknown sequence {seq_id}"),
            })?;
        let seq_len = entry.layer_lens.get(layer).copied().unwrap_or(0);
        if seq_len == 0 {
            return Ok(vec![0.0f32; self.config.head_dim]);
        }

        let scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let head_off = kv_head * self.config.head_dim;

        // Pass 1: compute scaled dot-product logits across physical pages
        let mut scores = Vec::with_capacity(seq_len);
        let mut max_score = f32::NEG_INFINITY;

        for pos in 0..seq_len {
            let logical_blk = pos / self.config.block_size;
            let in_blk = pos % self.config.block_size;
            let phys_id = entry
                .block_table
                .get(logical_blk)
                .copied()
                .ok_or(Error::InternalInvariant {
                    message: "paged_attention missing block",
                })?;
            let blk = self.blocks.get(phys_id).ok_or(Error::InternalInvariant {
                message: "paged_attention invalid physical block",
            })?;
            let k_start = layer * self.layer_stride + in_blk * self.kv_stride + head_off;
            let k_slice = blk
                .k
                .get(k_start..k_start + self.config.head_dim)
                .ok_or(Error::InternalInvariant {
                    message: "paged_attention k slice bounds",
                })?;
            let mut dot = 0.0f32;
            for (&q, &k) in q_head.iter().zip(k_slice.iter()) {
                dot += q * k;
            }
            let s = dot * scale;
            max_score = max_score.max(s);
            scores.push(s);
        }

        // Pass 2: numerically stable softmax
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        // Pass 3: weighted sum of V vectors across physical pages
        let mut out = vec![0.0f32; self.config.head_dim];
        for (pos, &w_raw) in scores.iter().enumerate() {
            let weight = w_raw * inv_sum;
            let logical_blk = pos / self.config.block_size;
            let in_blk = pos % self.config.block_size;
            let phys_id = entry
                .block_table
                .get(logical_blk)
                .copied()
                .ok_or(Error::InternalInvariant {
                    message: "paged_attention v missing block",
                })?;
            let blk = self.blocks.get(phys_id).ok_or(Error::InternalInvariant {
                message: "paged_attention v invalid physical block",
            })?;
            let v_start = layer * self.layer_stride + in_blk * self.kv_stride + head_off;
            let v_slice = blk
                .v
                .get(v_start..v_start + self.config.head_dim)
                .ok_or(Error::InternalInvariant {
                    message: "paged_attention v slice bounds",
                })?;
            for (o, &v) in out.iter_mut().zip(v_slice.iter()) {
                *o += weight * v;
            }
        }

        Ok(out)
    }

    /// Captures a [`PagedCacheStats`] snapshot.
    pub fn stats(&self) -> PagedCacheStats {
        let mut used_blocks = 0usize;
        let mut shared_blocks = 0usize;
        for blk in &self.blocks {
            if blk.ref_count > 0 {
                used_blocks += 1;
                if blk.ref_count > 1 {
                    shared_blocks += 1;
                }
            }
        }
        PagedCacheStats {
            total_blocks: self.config.num_blocks,
            used_blocks,
            free_blocks: self.free_list.len(),
            shared_blocks,
            active_sequences: self.sequences.len(),
        }
    }

    fn alloc_block(&mut self) -> Result<usize> {
        let blk_id = self
            .free_list
            .pop()
            .ok_or_else(|| Error::AllocationLimitExceeded {
                requested_bytes: self.config.num_blocks + 1,
                limit_bytes: self.config.num_blocks,
            })?;
        if let Some(blk) = self.blocks.get_mut(blk_id) {
            blk.ref_count = 1;
        }
        Ok(blk_id)
    }
}
