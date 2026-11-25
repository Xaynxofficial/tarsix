//! Verification tests for Low-Rank Adaptation (`LoRALinear`):
//! - Zero-init identity preservation at step 0
//! - Parameter-efficient fine-tuning convergence with frozen base weights
//! - Exact mathematical equivalence after `merge_weights()`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_autograd::Tape;
use as_core::DType;
use as_nn::{Linear, LoRALinear, Module, Rng, mse_loss};
use as_tensor::Tensor;

#[test]
fn lora_zero_init_training_and_exact_merge() {
    let mut rng = Rng::from_seed(42);
    let base = Linear::new("proj", 32, 32, true, DType::F32, &mut rng).unwrap();
    let mut lora = LoRALinear::from_linear("proj", &base, 4, 8.0, &mut rng).unwrap();

    // Only 25% of parameters are trainable for 32x32 rank 4 (for 4096x4096 rank 8 it's 0.39%)
    assert!((lora.trainable_param_ratio() - 0.25).abs() < 1e-6);

    let x_vals: Vec<f32> = (0..4 * 32).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let target_vals: Vec<f32> = (0..4 * 32).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let x_t = Tensor::from_f32(&x_vals, vec![4, 32]).unwrap();
    let target_t = Tensor::from_f32(&target_vals, vec![4, 32]).unwrap();

    // 1. At step 0 (B == 0), LoRA output must match frozen base identically
    let mut t0 = Tape::new();
    let vx0 = t0.var(x_t.clone(), false);
    let (base_out, _) = base.forward(&mut t0, vx0).unwrap();
    let (lora_out, _) = lora.forward(&mut t0, vx0).unwrap();
    let base_vec = t0.value(base_out).unwrap().to_f32_vec().unwrap();
    let lora_vec = t0.value(lora_out).unwrap().to_f32_vec().unwrap();
    for (&a, &b) in base_vec.iter().zip(lora_vec.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // 2. Fine-tune ONLY lora_a and lora_b via simple SGD steps
    let base_snapshot = lora.base_weight().to_f32_vec().unwrap();
    let mut first_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let lr = Tensor::full(vec![], 0.2, DType::F32).unwrap();

    for step in 0..40 {
        let mut tape = Tape::new();
        let vx = tape.var(x_t.clone(), false);
        let vt = tape.var(target_t.clone(), false);
        let (pred, pvars) = lora.forward(&mut tape, vx).unwrap();
        let loss = mse_loss(&mut tape, pred, vt).unwrap();
        let lv = tape.value(loss).unwrap().to_f32_vec().unwrap()[0];
        if step == 0 {
            first_loss = lv;
        }
        final_loss = lv;

        let grads = tape.backward(loss).unwrap();
        let mut params = lora.parameters_mut();
        for (p, &v) in params.iter_mut().zip(pvars.iter()) {
            let g = grads.get(v).unwrap();
            let updated = p.data().sub(&g.mul(&lr).unwrap()).unwrap();
            p.set_data(updated);
        }
    }

    assert!(
        final_loss < first_loss * 0.5,
        "LoRA failed to reduce loss: first={first_loss}, final={final_loss}"
    );
    // Base weight must remain completely frozen/unchanged
    assert_eq!(lora.base_weight().to_f32_vec().unwrap(), base_snapshot);

    // 3. Merge LoRA weights into standard Linear and verify identical inference output
    let merged = lora.merge_weights().unwrap();
    let mut eval_tape = Tape::new();
    eval_tape.set_grad_enabled(false);
    let vx = eval_tape.var(x_t, false);
    let (unmerged_y, _) = lora.forward(&mut eval_tape, vx).unwrap();
    let (merged_y, _) = merged.forward(&mut eval_tape, vx).unwrap();

    let u_vec = eval_tape.value(unmerged_y).unwrap().to_f32_vec().unwrap();
    let m_vec = eval_tape.value(merged_y).unwrap().to_f32_vec().unwrap();
    for (&u, &m) in u_vec.iter().zip(m_vec.iter()) {
        assert!((u - m).abs() < 1e-4, "merged mismatch: {u} vs {m}");
    }
}
