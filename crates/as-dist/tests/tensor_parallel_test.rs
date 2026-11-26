//! Integration test for Megatron-LM Tensor Parallelism (`ColumnParallelLinear`,
//! `RowParallelLinear`, `TensorParallelMlp`) verified against single-device reference MLP.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::thread;

use as_dist::{LocalCluster, TensorParallelMlp};
use as_tensor::Tensor;

#[test]
fn tensor_parallel_mlp_4_way_matches_single_device_reference_exactly() {
    let batch = 3;
    let d_in = 4;
    let d_hidden = 8; // divisible by 4
    let d_out = 4;

    let x_vals: Vec<f32> = (0..batch * d_in).map(|i| ((i as f32) * 0.17).sin()).collect();
    let w1_vals: Vec<f32> = (0..d_hidden * d_in).map(|i| ((i as f32) * 0.11).cos()).collect();
    let b1_vals: Vec<f32> = (0..d_hidden).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let w2_vals: Vec<f32> = (0..d_out * d_hidden).map(|i| ((i as f32) * 0.09).sin()).collect();
    let b2_vals: Vec<f32> = (0..d_out).map(|i| 0.2 - (i as f32) * 0.05).collect();

    let x = Tensor::from_f32(&x_vals, vec![batch, d_in]).unwrap();
    let w1 = Tensor::from_f32(&w1_vals, vec![d_hidden, d_in]).unwrap();
    let b1 = Tensor::from_f32(&b1_vals, vec![1, d_hidden]).unwrap();
    let w2 = Tensor::from_f32(&w2_vals, vec![d_out, d_hidden]).unwrap();
    let b2 = Tensor::from_f32(&b2_vals, vec![1, d_out]).unwrap();

    // Single-device reference: ReLU(X @ W1^T + b1) @ W2^T + b2
    let ref_h = x.matmul(&w1.transpose(0, 1).unwrap()).unwrap().add(&b1).unwrap().relu().unwrap();
    let ref_out = ref_h
        .matmul(&w2.transpose(0, 1).unwrap())
        .unwrap()
        .add(&b2)
        .unwrap()
        .to_f32_vec()
        .unwrap();

    // 4-way Tensor Parallel execution across 4 workers
    let world_size = 4;
    let comms = LocalCluster::create(world_size).unwrap();
    let handles: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(rank, comm)| {
            let x_c = x.clone();
            let w1_c = w1.clone();
            let b1_c = b1.clone();
            let w2_c = w2.clone();
            let b2_c = b2.clone();
            thread::spawn(move || {
                let tp = TensorParallelMlp::from_full_weights(
                    &w1_c,
                    Some(&b1_c),
                    &w2_c,
                    Some(&b2_c),
                    rank,
                    world_size,
                )
                .unwrap();
                tp.forward(&x_c, &comm).unwrap().to_f32_vec().unwrap()
            })
        })
        .collect();

    for h in handles {
        let rank_out = h.join().unwrap();
        assert_eq!(rank_out.len(), ref_out.len());
        for (e, a) in ref_out.iter().zip(rank_out.iter()) {
            assert!((e - a).abs() < 1e-4, "TP mismatch: expected {e}, got {a}");
        }
    }
}
