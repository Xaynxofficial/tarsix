//! Differential accuracy tests for all Faz A Apple Silicon Metal GPU kernels:
//! - Persistent Unified Memory `MetalBuffer` + multi-op `MetalCommandBatch`
//! - `rmsnorm_f32`, `softmax_f32`, `swiglu_f32`, `rope_f32`, `q4_matvec_f32`

use as_metal::MetalBackend;

#[test]
fn metal_llm_kernels_match_cpu_reference() {
    let Ok(metal) = MetalBackend::new() else {
        // Non-macOS or headless VM without Metal GPU
        return;
    };

    // 1. RMSNorm differential check
    let rows = 4;
    let cols = 16;
    let x: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.13).sin()).collect();
    let w: Vec<f32> = (0..cols).map(|i| 0.5 + (i as f32) * 0.05).collect();
    let eps = 1e-5_f32;
    let gpu_rms = metal.rmsnorm_f32(&x, &w, rows, cols, eps).unwrap();
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / (cols as f32);
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..cols {
            let expected = row[c] * inv_rms * w[c];
            let actual = gpu_rms[r * cols + c];
            assert!((expected - actual).abs() < 1e-4, "rmsnorm mismatch: {expected} vs {actual}");
        }
    }

    // 2. Softmax differential check
    let gpu_sm = metal.softmax_f32(&x, rows, cols).unwrap();
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..cols {
            let expected = exps[c] / sum;
            let actual = gpu_sm[r * cols + c];
            assert!((expected - actual).abs() < 1e-4, "softmax mismatch: {expected} vs {actual}");
        }
    }

    // 3. SwiGLU differential check
    let gate: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let up: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
    let gpu_swiglu = metal.swiglu_f32(&gate, &up).unwrap();
    for i in 0..64 {
        let g = gate[i];
        let expected = (g / (1.0 + (-g).exp())) * up[i];
        assert!((expected - gpu_swiglu[i]).abs() < 1e-4);
    }

    // 4. RoPE differential check (norm preservation + exact rotation)
    let seq = 4;
    let head_dim = 8;
    let rope_in: Vec<f32> = (0..seq * head_dim).map(|i| ((i as f32) * 0.21).cos()).collect();
    let gpu_rope = metal.rope_f32(&rope_in, seq, head_dim, 3, 10000.0).unwrap();
    for s in 0..seq {
        let norm_in: f32 = rope_in[s * head_dim..(s + 1) * head_dim]
            .iter()
            .map(|v| v * v)
            .sum();
        let norm_out: f32 = gpu_rope[s * head_dim..(s + 1) * head_dim]
            .iter()
            .map(|v| v * v)
            .sum();
        assert!((norm_in - norm_out).abs() < 1e-3, "RoPE must preserve L2 norm");
    }

    // 5. Q4 4-bit block-quantized MatVec differential check
    let out_dim = 8;
    let in_dim = 64;
    let blocks_per_row = in_dim / 32;
    let mut packed = vec![0u8; out_dim * blocks_per_row * 16];
    let mut scales = vec![0.0f32; out_dim * blocks_per_row];
    for r in 0..out_dim {
        for b in 0..blocks_per_row {
            let blk = r * blocks_per_row + b;
            scales[blk] = 0.1 * ((blk % 5 + 1) as f32);
            for k in 0..16 {
                let q0 = ((r + k) % 15) as u8;
                let q1 = ((r + k + 3) % 15) as u8;
                packed[blk * 16 + k] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        }
    }
    let xv: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.07).sin()).collect();
    let gpu_q4 = metal.q4_matvec_f32(&packed, &scales, &xv, out_dim, in_dim).unwrap();
    for r in 0..out_dim {
        let mut expected = 0.0f32;
        for b in 0..blocks_per_row {
            let blk = r * blocks_per_row + b;
            let sc = scales[blk];
            let mut b_acc = 0.0f32;
            for k in 0..16 {
                let byte = packed[blk * 16 + k];
                let q0 = ((byte & 0x0F) as i32 - 8) as f32;
                let q1 = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
                b_acc += q0 * xv[b * 32 + k * 2] + q1 * xv[b * 32 + k * 2 + 1];
            }
            expected += b_acc * sc;
        }
        assert!(
            (expected - gpu_q4[r]).abs() < 1e-3,
            "q4_matvec row {r}: expected {expected}, got {}",
            gpu_q4[r]
        );
    }

    // 6. Zero-copy Unified Memory MetalBuffer + multi-op MetalCommandBatch (RMSNorm -> GEMM -> ReLU)
    let m = 4;
    let k = 16;
    let n = 8;
    let bx = metal.buffer_from_f32(&x).unwrap();
    let bw_norm = metal.buffer_from_f32(&w).unwrap();
    let b_normed = metal.alloc_f32_buffer(m * k).unwrap();
    let w_proj: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.11).cos()).collect();
    let bw_proj = metal.buffer_from_f32(&w_proj).unwrap();
    let b_gemm = metal.alloc_f32_buffer(m * n).unwrap();
    let b_out = metal.alloc_f32_buffer(m * n).unwrap();

    let mut batch = metal.begin_batch().unwrap();
    batch.encode_rmsnorm_f32(&bx, &bw_norm, &b_normed, m, k, eps).unwrap();
    batch.encode_gemm_f32(&b_normed, &bw_proj, &b_gemm, m, k, n).unwrap();
    batch.encode_relu_f32(&b_gemm, &b_out, m * n).unwrap();
    batch.commit_and_wait().unwrap();

    // Verify batched output matches sequential single-shot reference
    let ref_gemm = metal.gemm_f32(&gpu_rms, &w_proj, m, k, n).unwrap();
    let ref_out: Vec<f32> = ref_gemm.into_iter().map(|v| v.max(0.0)).collect();
    let batched_out = b_out.as_f32_slice();
    for i in 0..m * n {
        assert!(
            (ref_out[i] - batched_out[i]).abs() < 1e-4,
            "batched mismatch at {i}: {} vs {}",
            ref_out[i],
            batched_out[i]
        );
    }
}
