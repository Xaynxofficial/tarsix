//! Comprehensive numerical correctness, broadcasting, layout, and security tests.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_tensor::{BF16, CpuBackend, DType, Error, F16, MAX_RANK, MemoryLimits, SliceRange, Tensor};

fn assert_close_f32(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol,
            "index {i}: actual={a}, expected={e}, diff={diff} > tol={tol}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol,
            "index {i}: actual={a}, expected={e}, diff={diff} > tol={tol}"
        );
    }
}

/// Independent naive reference matmul for differential numerical verification.
fn naive_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            c[i * n + j] = sum as f32;
        }
    }
    c
}

#[test]
fn tiled_matmul_matches_naive_reference_non_tile_aligned() {
    // Dimensions intentionally not multiples of TILE=64 to exercise all boundary tiles.
    let (m, k, n) = (73, 131, 67);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.125)
        .collect();

    let a = Tensor::from_f32(&a_data, vec![m, k]).unwrap();
    let b = Tensor::from_f32(&b_data, vec![k, n]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.dims(), &[m, n]);
    let expected = naive_matmul_f32(&a_data, &b_data, m, k, n);
    assert_close_f32(&c.to_f32_vec().unwrap(), &expected, 1e-4, 1e-5);
}

#[test]
fn batched_broadcast_matmul_and_transposed_views() {
    // [2, 1, 3, 4] @ [3, 4, 2] -> [2, 3, 3, 2]
    let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5).collect();
    let b_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let a = Tensor::from_f32(&a_data, vec![2, 1, 3, 4]).unwrap();
    let b = Tensor::from_f32(&b_data, vec![3, 4, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3, 3, 2]);

    // Transposed 2D matmul: (A^T) @ B
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let xt = x.transpose(0, 1).unwrap(); // [3, 2]
    assert!(!xt.is_contiguous());
    let gram = xt.matmul(&x).unwrap(); // [3, 3]
    assert_eq!(gram.dims(), &[3, 3]);
    assert_close_f32(
        &gram.to_f32_vec().unwrap(),
        &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0],
        1e-6,
        1e-6,
    );
}

#[test]
fn broadcast_arithmetic_and_operators() {
    let a = Tensor::from_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let bias = Tensor::from_f64(&[10.0, 20.0, 30.0], vec![1, 3]).unwrap();

    let sum = (&a + &bias).unwrap();
    assert_eq!(sum.dims(), &[2, 3]);
    assert_close_f64(
        &sum.to_f64_vec().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        1e-12,
        1e-12,
    );

    // Scalar broadcast [2, 3] * []
    let scale = Tensor::from_f64(&[2.5], vec![]).unwrap();
    let scaled = (&a * &scale).unwrap();
    assert_close_f64(
        &scaled.to_f64_vec().unwrap(),
        &[2.5, 5.0, 7.5, 10.0, 12.5, 15.0],
        1e-12,
        1e-12,
    );
}

#[test]
fn f16_and_bf16_arithmetic_and_matmul() {
    let a_f32 = [1.0f32, 2.0, 3.0, 4.0];
    let b_f32 = [5.0f32, 6.0, 7.0, 8.0];

    let a16: Vec<F16> = a_f32.iter().map(|&x| F16::from_f32(x)).collect();
    let b16: Vec<F16> = b_f32.iter().map(|&x| F16::from_f32(x)).collect();
    let ta = Tensor::from_f16(&a16, vec![2, 2]).unwrap();
    let tb = Tensor::from_f16(&b16, vec![2, 2]).unwrap();
    let tc = ta.matmul(&tb).unwrap();
    let out: Vec<f32> = tc.to_f16_vec().unwrap().into_iter().map(F16::to_f32).collect();
    assert_close_f32(&out, &[19.0, 22.0, 43.0, 50.0], 1e-2, 1e-2);

    let abf: Vec<BF16> = a_f32.iter().map(|&x| BF16::from_f32(x)).collect();
    let bbf: Vec<BF16> = b_f32.iter().map(|&x| BF16::from_f32(x)).collect();
    let tba = Tensor::from_bf16(&abf, vec![2, 2]).unwrap();
    let tbb = Tensor::from_bf16(&bbf, vec![2, 2]).unwrap();
    let tbc = tba.matmul(&tbb).unwrap();
    let out_bf: Vec<f32> = tbc
        .to_bf16_vec()
        .unwrap()
        .into_iter()
        .map(BF16::to_f32)
        .collect();
    assert_close_f32(&out_bf, &[19.0, 22.0, 43.0, 50.0], 1e-1, 1e-2);
}

#[test]
fn reductions_global_and_axis_with_keepdim() {
    let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    assert_close_f32(&t.sum().unwrap().to_f32_vec().unwrap(), &[21.0], 1e-6, 1e-6);
    assert_close_f32(&t.mean().unwrap().to_f32_vec().unwrap(), &[3.5], 1e-6, 1e-6);
    assert_close_f32(&t.max().unwrap().to_f32_vec().unwrap(), &[6.0], 1e-6, 1e-6);
    assert_close_f32(&t.min().unwrap().to_f32_vec().unwrap(), &[1.0], 1e-6, 1e-6);

    let s0 = t.sum_axis(0, false).unwrap();
    assert_eq!(s0.dims(), &[3]);
    assert_close_f32(&s0.to_f32_vec().unwrap(), &[5.0, 7.0, 9.0], 1e-6, 1e-6);

    let m1 = t.mean_axis(1, true).unwrap();
    assert_eq!(m1.dims(), &[2, 1]);
    assert_close_f32(&m1.to_f32_vec().unwrap(), &[2.0, 5.0], 1e-6, 1e-6);
}

#[test]
fn slicing_permute_reshape_and_unary_ops() {
    let t = Tensor::arange_f32(0.0, 12.0, 1.0)
        .unwrap()
        .reshape(vec![3, 4])
        .unwrap();
    let sliced = t
        .slice(&[SliceRange::range(1, 3), SliceRange::stepped(0, 4, 2)])
        .unwrap();
    assert_eq!(sliced.dims(), &[2, 2]);
    assert_close_f32(&sliced.to_f32_vec().unwrap(), &[4.0, 6.0, 8.0, 10.0], 1e-6, 1e-6);

    // Non-contiguous reshape automatically materializes
    let flat = sliced.reshape(vec![4]).unwrap();
    assert_close_f32(&flat.to_f32_vec().unwrap(), &[4.0, 6.0, 8.0, 10.0], 1e-6, 1e-6);

    let neg = Tensor::from_f32(&[-2.0, 0.0, 4.0], vec![3]).unwrap();
    assert_close_f32(&neg.relu().unwrap().to_f32_vec().unwrap(), &[0.0, 0.0, 4.0], 1e-6, 1e-6);
    assert_close_f32(&neg.abs().unwrap().to_f32_vec().unwrap(), &[2.0, 0.0, 4.0], 1e-6, 1e-6);
}

#[test]
fn security_limits_and_error_handling() {
    // 1. OOM / allocation limit guard
    let tight_backend = CpuBackend::with_limits(MemoryLimits::new(64, 4));
    let err = Tensor::zeros_on(vec![100], DType::F32, tight_backend).unwrap_err();
    assert!(matches!(err, Error::AllocationLimitExceeded { .. }));

    // 2. Rank ceiling guard
    let err = Tensor::zeros(vec![1; MAX_RANK + 1], DType::F32).unwrap_err();
    assert!(matches!(err, Error::InvalidRank { .. }));

    // 3. Integer overflow protection in i32 arithmetic
    let max_i = Tensor::from_i32(&[i32::MAX], vec![1]).unwrap();
    let one_i = Tensor::from_i32(&[1], vec![1]).unwrap();
    assert!(matches!(
        max_i.add(&one_i),
        Err(Error::ArithmeticOverflow { .. })
    ));

    // 4. Integer division by zero returns Error instead of panicking
    let zero_i = Tensor::from_i32(&[0], vec![1]).unwrap();
    assert!(matches!(
        one_i.div(&zero_i),
        Err(Error::InvalidArgument { .. })
    ));

    // 5. Unsupported dtype op (exp on I32)
    assert!(matches!(
        one_i.exp(),
        Err(Error::UnsupportedDType {
            op: "exp",
            dtype: DType::I32
        })
    ));

    // 6. Dtype mismatch in binary op
    let f = Tensor::zeros(vec![1], DType::F32).unwrap();
    assert!(matches!(f.add(&one_i), Err(Error::DTypeMismatch { .. })));
}
