//! Integration tests for `as-autograd`: reverse-mode VJP, broadcast unbroadcasting,
//! gradient accumulation, gradient checkpointing, forward-mode JVP, and finite-difference gradcheck.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_autograd::{DualTensor, Gradients, Tape, numerical_gradient, verify_gradients};
use as_tensor::Tensor;

#[test]
fn gradcheck_mlp_block_with_broadcast_and_activations() {
    // f(x, w, b) = mean( exp( relu(x @ w + b) * 0.25 ) ) + sum( x^2 ) * 0.1
    let x_t = Tensor::from_f64(&[0.5, -0.3, 1.2, 0.8, 0.4, -0.7], vec![2, 3]).unwrap();
    let w_t = Tensor::from_f64(&[0.4, -0.2, 0.6, 0.9, -0.5, 0.3], vec![3, 2]).unwrap();
    let b_t = Tensor::from_f64(&[0.2, 0.5], vec![1, 2]).unwrap();

    let eval_scalar = |x: &Tensor, w: &Tensor, b: &Tensor| -> as_core::Result<f64> {
        let mut tape = Tape::new();
        let vx = tape.var(x.clone(), false);
        let vw = tape.var(w.clone(), false);
        let vb = tape.var(b.clone(), false);
        let scale = tape.var(Tensor::from_f64(&[0.25], vec![])?, false);
        let reg_scale = tape.var(Tensor::from_f64(&[0.1], vec![])?, false);

        let mm = tape.matmul(vx, vw)?;
        let lin = tape.add(mm, vb)?;
        let act = tape.relu(lin)?;
        let scaled = tape.mul(act, scale)?;
        let e = tape.exp(scaled)?;
        let term1 = tape.mean(e)?;

        let x_sq = tape.powf(vx, 2.0)?;
        let x_sum = tape.sum(x_sq)?;
        let term2 = tape.mul(x_sum, reg_scale)?;
        let loss = tape.add(term1, term2)?;
        Ok(tape.value(loss)?.to_f64_vec()?[0])
    };

    // Analytical gradients via Tape
    let mut tape = Tape::new();
    let vx = tape.var(x_t.clone(), true);
    let vw = tape.var(w_t.clone(), true);
    let vb = tape.var(b_t.clone(), true);
    let scale = tape.var(Tensor::from_f64(&[0.25], vec![]).unwrap(), false);
    let reg_scale = tape.var(Tensor::from_f64(&[0.1], vec![]).unwrap(), false);

    let mm = tape.matmul(vx, vw).unwrap();
    let lin = tape.add(mm, vb).unwrap();
    let act = tape.relu(lin).unwrap();
    let scaled = tape.mul(act, scale).unwrap();
    let e = tape.exp(scaled).unwrap();
    let term1 = tape.mean(e).unwrap();
    let x_sq = tape.powf(vx, 2.0).unwrap();
    let x_sum = tape.sum(x_sq).unwrap();
    let term2 = tape.mul(x_sum, reg_scale).unwrap();
    let loss = tape.add(term1, term2).unwrap();

    let grads = tape.backward(loss).unwrap();

    // Verify dL/dx, dL/dw, dL/db against central finite differences
    let num_gx = numerical_gradient(&x_t, 1e-6, |x| eval_scalar(x, &w_t, &b_t)).unwrap();
    let num_gw = numerical_gradient(&w_t, 1e-6, |w| eval_scalar(&x_t, w, &b_t)).unwrap();
    let num_gb = numerical_gradient(&b_t, 1e-6, |b| eval_scalar(&x_t, &w_t, b)).unwrap();

    let rx = verify_gradients(grads.get(vx).unwrap(), &num_gx, 1e-6, 1e-5).unwrap();
    let rw = verify_gradients(grads.get(vw).unwrap(), &num_gw, 1e-6, 1e-5).unwrap();
    let rb = verify_gradients(grads.get(vb).unwrap(), &num_gb, 1e-6, 1e-5).unwrap();

    assert!(rx.passed, "dL/dx gradcheck failed: {rx:?}");
    assert!(rw.passed, "dL/dw gradcheck failed: {rw:?}");
    assert!(rb.passed, "dL/db gradcheck failed: {rb:?}");
}

#[test]
fn gradient_checkpointing_matches_uncheckpointed_exactly() {
    let x_t = Tensor::from_f64(&[1.2, 0.7, 2.1, 0.4], vec![2, 2]).unwrap();
    let w_t = Tensor::from_f64(&[0.5, 0.8, 0.3, 1.1], vec![2, 2]).unwrap();

    // Standard uncheckpointed pass
    let mut t1 = Tape::new();
    let x1 = t1.var(x_t.clone(), true);
    let w1 = t1.var(w_t.clone(), true);
    let h1 = t1.matmul(x1, w1).unwrap();
    let h2 = t1.sqrt(h1).unwrap();
    let h3 = t1.ln(h2).unwrap();
    let loss1 = t1.sum(h3).unwrap();
    let g1 = t1.backward(loss1).unwrap();

    // Checkpointed pass (recomputes matmul -> sqrt -> ln during backward)
    let mut t2 = Tape::new();
    let x2 = t2.var(x_t, true);
    let w2 = t2.var(w_t, true);
    let cp = t2
        .checkpoint(&[x2, w2], |sub, ins| {
            let h = sub.matmul(ins[0], ins[1])?;
            let s = sub.sqrt(h)?;
            sub.ln(s)
        })
        .unwrap();
    let loss2 = t2.sum(cp).unwrap();
    let g2 = t2.backward(loss2).unwrap();

    let r_x = verify_gradients(g1.get(x1).unwrap(), g2.get(x2).unwrap(), 1e-12, 1e-12).unwrap();
    let r_w = verify_gradients(g1.get(w1).unwrap(), g2.get(w2).unwrap(), 1e-12, 1e-12).unwrap();
    assert!(r_x.passed, "checkpoint dx mismatch: {r_x:?}");
    assert!(r_w.passed, "checkpoint dw mismatch: {r_w:?}");
}

#[test]
fn forward_mode_jvp_matches_reverse_mode_vjp_inner_product() {
    // Identity: <v, J(x) u> == <J(x)^T v, u>
    let x = Tensor::from_f64(&[1.1, 0.6, 1.7, 0.9], vec![2, 2]).unwrap();
    let u = Tensor::from_f64(&[0.3, -0.5, 0.8, 0.2], vec![2, 2]).unwrap();
    let v = Tensor::from_f64(&[-0.4, 0.7, 0.2, 1.3], vec![2, 2]).unwrap();

    // Forward-mode JVP: f(x) = exp(x @ x^T) / (x + 2)
    let two = DualTensor::constant(Tensor::from_f64(&[2.0], vec![]).unwrap()).unwrap();
    let dx = DualTensor::new(x.clone(), u.clone()).unwrap();
    let xt = dx.transpose(0, 1).unwrap();
    let num = dx.matmul(&xt).unwrap().exp().unwrap();
    let den = dx.add(&two).unwrap();
    let y_dual = num.div(&den).unwrap();
    let jvp = y_dual.tangent();

    // <v, J u>
    let lhs_inner = v.mul(jvp).unwrap().sum().unwrap().to_f64_vec().unwrap()[0];

    // Reverse-mode VJP with seed v
    let mut tape = Tape::new();
    let vx = tape.var(x, true);
    let vtwo = tape.var(Tensor::from_f64(&[2.0], vec![]).unwrap(), false);
    let vxt = tape.transpose(vx, 0, 1).unwrap();
    let vnum = tape.matmul(vx, vxt).unwrap();
    let vexp = tape.exp(vnum).unwrap();
    let vden = tape.add(vx, vtwo).unwrap();
    let vy = tape.div(vexp, vden).unwrap();

    let mut grads = Gradients::new();
    tape.backward_with_seed_into(vy, v, &mut grads).unwrap();
    let vjp = grads.get(vx).unwrap();

    // <J^T v, u>
    let rhs_inner = vjp.mul(&u).unwrap().sum().unwrap().to_f64_vec().unwrap()[0];
    assert!(
        (lhs_inner - rhs_inner).abs() < 1e-10,
        "JVP/VJP adjoint identity failed: {lhs_inner} vs {rhs_inner}"
    );
}

#[test]
fn gradient_accumulation_across_microbatches() {
    let mut tape = Tape::new();
    let x = tape.var(Tensor::from_f64(&[2.0, 3.0], vec![2]).unwrap(), true);
    let l1 = tape.sum(x).unwrap(); // d/dx = [1, 1]
    let x2 = tape.powf(x, 2.0).unwrap();
    let l2 = tape.sum(x2).unwrap(); // d/dx = [4, 6]

    let mut acc = Gradients::new();
    tape.backward_into(l1, &mut acc).unwrap();
    tape.backward_into(l2, &mut acc).unwrap();

    let g = acc.get(x).unwrap().to_f64_vec().unwrap();
    assert!((g[0] - 5.0).abs() < 1e-12);
    assert!((g[1] - 7.0).abs() < 1e-12);
}
