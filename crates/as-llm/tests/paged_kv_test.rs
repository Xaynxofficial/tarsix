//! Integration tests for Faz B `PagedKvCache` (PagedAttention + CoW prefix sharing).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_llm::{PagedKvCache, PagedKvConfig};

#[test]
fn paged_kv_cache_cow_prefix_sharing_and_exact_paged_attention() {
    let cfg = PagedKvConfig {
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
        block_size: 4,
        num_blocks: 8,
    };
    let mut cache = PagedKvCache::new(cfg).unwrap();
    cache.allocate_sequence(1).unwrap();

    // Append a 4-token shared system prompt (fills exactly 1 physical block)
    let mut ref_k = Vec::new();
    let mut ref_v = Vec::new();
    for t in 0..4 {
        let k: Vec<f32> = (0..8).map(|i| ((t * 8 + i) as f32) * 0.1).collect();
        let v: Vec<f32> = (0..8).map(|i| ((t * 8 + i) as f32) * 0.05 + 1.0).collect();
        cache.append_token_kv(1, 0, &k, &v).unwrap();
        cache.append_token_kv(1, 1, &k, &v).unwrap();
        ref_k.push(k);
        ref_v.push(v);
    }
    assert_eq!(cache.stats().used_blocks, 1);
    assert_eq!(cache.stats().shared_blocks, 0);

    // Zero-copy fork sequence 1 -> sequence 2 (shared system prompt uses 0 extra blocks!)
    cache.fork_sequence(1, 2).unwrap();
    assert_eq!(cache.stats().used_blocks, 1);
    assert_eq!(cache.stats().shared_blocks, 1);

    // Sequence 2 diverges by appending token 5 (crosses block boundary -> allocates 1 new block, block 0 stays shared!)
    let k5 = vec![0.5f32; 8];
    let v5 = vec![2.0f32; 8];
    cache.append_token_kv(2, 0, &k5, &v5).unwrap();
    cache.append_token_kv(2, 1, &k5, &v5).unwrap();
    assert_eq!(cache.stats().used_blocks, 2);
    assert_eq!(cache.stats().shared_blocks, 1);

    // Verify PagedAttention on seq 1 matches exact contiguous attention
    let q = vec![0.2, -0.1, 0.4, 0.3];
    let paged_out = cache.paged_attention(1, 0, 0, &q).unwrap();

    let scale = 1.0 / (4.0f32).sqrt();
    let scores: Vec<f32> = ref_k
        .iter()
        .map(|k| {
            q.iter()
                .zip(k[0..4].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale
        })
        .collect();
    let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut expected = vec![0.0f32; 4];
    for (t, &e) in exps.iter().enumerate() {
        let w = e / sum;
        for d in 0..4 {
            expected[d] += w * ref_v[t][d];
        }
    }
    for d in 0..4 {
        assert!((expected[d] - paged_out[d]).abs() < 1e-5);
    }

    // Freeing both sequences returns all blocks to the free list
    cache.free_sequence(1).unwrap();
    assert_eq!(cache.stats().used_blocks, 2); // seq 2 still holds block 0 & block 1
    assert_eq!(cache.stats().shared_blocks, 0);
    cache.free_sequence(2).unwrap();
    assert_eq!(cache.stats().used_blocks, 0);
    assert_eq!(cache.stats().free_blocks, 8);
}
