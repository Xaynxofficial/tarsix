//! Reproducible, zero-dependency benchmark runner for Milestone 1 CPU Tensor Engine.
//!
//! Run with: `cargo run --release -p as-bench`

use std::hint::black_box;
use std::time::{Duration, Instant};

use as_tensor::{DType, Result, Tensor};

struct BenchStats {
    name: String,
    median_ms: f64,
    mean_ms: f64,
    p95_ms: f64,
    metric_label: String,
    metric_value: f64,
}

fn run_bench<F>(
    name: &str,
    warmup: usize,
    iters: usize,
    metric_label: &str,
    ops_per_iter: f64,
    mut f: F,
) -> Result<BenchStats>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..warmup {
        f()?;
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed());
    }

    samples.sort_unstable();
    let total_secs: f64 = samples.iter().map(Duration::as_secs_f64).sum();
    let mean_secs = total_secs / (iters.max(1) as f64);
    let median_secs = samples
        .get(iters / 2)
        .map_or(mean_secs, Duration::as_secs_f64);
    let p95_idx = ((iters * 95) / 100).min(iters.saturating_sub(1));
    let p95_secs = samples
        .get(p95_idx)
        .map_or(median_secs, Duration::as_secs_f64);

    let metric_value = if median_secs > 0.0 {
        ops_per_iter / median_secs
    } else {
        0.0
    };

    Ok(BenchStats {
        name: name.to_owned(),
        median_ms: median_secs * 1e3,
        mean_ms: mean_secs * 1e3,
        p95_ms: p95_secs * 1e3,
        metric_label: metric_label.to_owned(),
        metric_value,
    })
}

fn main() -> Result<()> {
    println!("=== Independent AI Computing Infrastructure — Milestone 1 CPU Benchmarks ===");
    println!(
        "Arch: {} | OS: {} | Build: {}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        if cfg!(debug_assertions) {
            "debug (WARNING: use --release)"
        } else {
            "release"
        }
    );
    println!(
        "{:<34} {:>10} {:>10} {:>10} {:>16}",
        "Benchmark", "Median(ms)", "Mean(ms)", "P95(ms)", "Throughput"
    );
    println!("{}", "-".repeat(84));

    let mut results = Vec::new();

    // 1. Element-wise contiguous add (1M f32 = 4 MB read + 4 MB read + 4 MB write = 12 MB)
    let n_elem = 1_000_000usize;
    let a_vec = Tensor::full(vec![n_elem], 1.25, DType::F32)?;
    let b_vec = Tensor::full(vec![n_elem], 2.75, DType::F32)?;
    let bytes_gb = (3.0 * (n_elem as f64) * 4.0) / 1e9;
    results.push(run_bench(
        "add_f32_contiguous_1M",
        5,
        25,
        "GB/s",
        bytes_gb,
        || {
            let out = black_box(&a_vec).add(black_box(&b_vec))?;
            black_box(out);
            Ok(())
        },
    )?);

    // 2. Broadcast add ([1024, 1024] + [1, 1024])
    let mat_1k = Tensor::full(vec![1024, 1024], 1.5, DType::F32)?;
    let bias_1k = Tensor::full(vec![1, 1024], 0.5, DType::F32)?;
    let bcast_gb = (2.0 * 1024.0 * 1024.0 * 4.0) / 1e9;
    results.push(run_bench(
        "add_f32_broadcast_1024x1024",
        3,
        15,
        "GB/s",
        bcast_gb,
        || {
            let out = black_box(&mat_1k).add(black_box(&bias_1k))?;
            black_box(out);
            Ok(())
        },
    )?);

    // 3. MatMul F32 256x256 (2 * N^3 FLOPs)
    let m256_a = Tensor::full(vec![256, 256], 0.5, DType::F32)?;
    let m256_b = Tensor::full(vec![256, 256], 0.25, DType::F32)?;
    let gflops_256 = (2.0 * 256.0 * 256.0 * 256.0) / 1e9;
    results.push(run_bench(
        "matmul_f32_256x256",
        3,
        15,
        "GFLOP/s",
        gflops_256,
        || {
            let out = black_box(&m256_a).matmul(black_box(&m256_b))?;
            black_box(out);
            Ok(())
        },
    )?);

    // 4. MatMul F32 512x512 (2 * N^3 FLOPs)
    let m512_a = Tensor::full(vec![512, 512], 0.5, DType::F32)?;
    let m512_b = Tensor::full(vec![512, 512], 0.25, DType::F32)?;
    let gflops_512 = (2.0 * 512.0 * 512.0 * 512.0) / 1e9;
    results.push(run_bench(
        "matmul_f32_512x512",
        2,
        7,
        "GFLOP/s",
        gflops_512,
        || {
            let out = black_box(&m512_a).matmul(black_box(&m512_b))?;
            black_box(out);
            Ok(())
        },
    )?);

    // 5. Reduction sum_axis on [1024, 1024]
    let red_gb = (1024.0 * 1024.0 * 4.0) / 1e9;
    results.push(run_bench(
        "sum_axis1_f32_1024x1024",
        3,
        15,
        "GB/s",
        red_gb,
        || {
            let out = black_box(&mat_1k).sum_axis(1, false)?;
            black_box(out);
            Ok(())
        },
    )?);

    for r in &results {
        println!(
            "{:<34} {:>10.3} {:>10.3} {:>10.3} {:>9.2} {}",
            r.name, r.median_ms, r.mean_ms, r.p95_ms, r.metric_value, r.metric_label
        );
    }

    Ok(())
}
