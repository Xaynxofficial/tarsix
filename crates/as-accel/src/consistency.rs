//! Differential numerical consistency test harness comparing any two [`Backend`]s.

use as_backend::Backend;
use as_core::Result;
use as_tensor::Tensor;

/// Summary report from differential cross-backend numerical verification.
#[derive(Debug, Clone)]
pub struct ConsistencyReport {
    /// Maximum absolute error observed across all checked operations.
    pub max_abs_error: f64,
    /// Number of operations compared.
    pub ops_checked: usize,
    /// Whether all outputs matched within `(atol, rtol)`.
    pub passed: bool,
}

/// Runs a canonical workload suite on `b1` (reference) and `b2` (candidate) and verifies
/// numerical consistency within `(atol, rtol)`.
pub fn verify_backend_consistency<B1: Backend, B2: Backend>(
    b1: &B1,
    b2: &B2,
    atol: f64,
    rtol: f64,
) -> Result<ConsistencyReport> {
    let mut max_abs_error = 0.0f64;
    let mut passed = true;
    let mut ops_checked = 0usize;

    let mut compare = |v1: &[f32], v2: &[f32]| {
        ops_checked += 1;
        if v1.len() != v2.len() {
            passed = false;
            return;
        }
        for (&a, &b) in v1.iter().zip(v2.iter()) {
            let diff = f64::from((a - b).abs());
            if diff > max_abs_error {
                max_abs_error = diff;
            }
            if diff > atol + rtol * f64::from(a.abs()) || a.is_nan() || b.is_nan() {
                passed = false;
            }
        }
    };

    // 1. MatMul (96x80 @ 80x72) — exercises multi-core parallel band path
    let a_data: Vec<f32> = (0..96 * 80).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
    let b_data: Vec<f32> = (0..80 * 72).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

    let a1 = Tensor::from_f32_on(&a_data, vec![96, 80], b1.clone())?;
    let b1_t = Tensor::from_f32_on(&b_data, vec![80, 72], b1.clone())?;
    let c1 = a1.matmul(&b1_t)?.to_f32_vec()?;

    let a2 = Tensor::from_f32_on(&a_data, vec![96, 80], b2.clone())?;
    let b2_t = Tensor::from_f32_on(&b_data, vec![80, 72], b2.clone())?;
    let c2 = a2.matmul(&b2_t)?.to_f32_vec()?;
    compare(&c1, &c2);

    // 2. Broadcast Row Add ([96, 72] + [1, 72])
    let bias_data: Vec<f32> = (0..72).map(|i| (i as f32) * 0.05).collect();
    let bias1 = Tensor::from_f32_on(&bias_data, vec![1, 72], b1.clone())?;
    let bias2 = Tensor::from_f32_on(&bias_data, vec![1, 72], b2.clone())?;
    let add1 = a1.matmul(&b1_t)?.add(&bias1)?.to_f32_vec()?;
    let add2 = a2.matmul(&b2_t)?.add(&bias2)?.to_f32_vec()?;
    compare(&add1, &add2);

    // 3. Axis reduction (sum_axis 1)
    let r1 = a1.sum_axis(1, false)?.to_f32_vec()?;
    let r2 = a2.sum_axis(1, false)?.to_f32_vec()?;
    compare(&r1, &r2);

    Ok(ConsistencyReport {
        max_abs_error,
        ops_checked,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use as_cpu::CpuBackend;

    use super::*;
    use crate::kernels::{CUDA_GEMM_F32_KERNEL, METAL_GEMM_F32_MSL, WGSL_GEMM_F32};
    use crate::registry::discover_backends;

    #[test]
    fn backend_discovery_kernels_and_consistency() {
        let backends = discover_backends();
        assert!(!backends.is_empty());
        assert!(backends[0].ready_for_execution);

        assert!(METAL_GEMM_F32_MSL.contains("gemm_f32_tiled"));
        assert!(WGSL_GEMM_F32.contains("@workgroup_size(16, 16, 1)"));
        assert!(CUDA_GEMM_F32_KERNEL.contains("__global__"));

        let cpu1 = CpuBackend::new();
        let cpu2 = CpuBackend::new();
        let report = verify_backend_consistency(&cpu1, &cpu2, 1e-6, 1e-6).unwrap();
        assert!(report.passed);
        assert_eq!(report.ops_checked, 3);
        assert!(report.max_abs_error < 1e-6);
    }
}
