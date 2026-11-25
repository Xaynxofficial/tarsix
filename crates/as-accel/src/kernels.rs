//! Authoritative GPU compute kernel sources for Metal (MSL), WebGPU (WGSL), and CUDA C++.
//!
//! Kept in-tree and versioned alongside the CPU reference backend so every GPU
//! implementation shares identical indexing, tiling, and numerical conventions.

/// Apple Silicon Metal Shading Language (MSL) 16x16 threadgroup-tiled `f32` GEMM kernel.
pub const METAL_GEMM_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GemmDims {
    uint M;
    uint K;
    uint N;
};

kernel void gemm_f32_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmDims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE = 16;
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0f;

    uint num_tiles = (dims.K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; ++t) {
        uint a_col = t * TILE + tid.x;
        uint b_row = t * TILE + tid.y;
        As[tid.y][tid.x] = (row < dims.M && a_col < dims.K) ? A[row * dims.K + a_col] : 0.0f;
        Bs[tid.y][tid.x] = (b_row < dims.K && col < dims.N) ? B[b_row * dims.N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k) {
            acc += As[tid.y][k] * Bs[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
"#;

/// WebGPU Shading Language (WGSL) 16x16 workgroup-tiled `f32` GEMM compute shader.
pub const WGSL_GEMM_F32: &str = r#"
struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    var acc: f32 = 0.0;
    let numTiles = (dims.K + 15u) / 16u;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * 16u + lid.x;
        let bRow = t * 16u + lid.y;
        if (row < dims.M && aCol < dims.K) {
            tileA[lid.y][lid.x] = A[row * dims.K + aCol];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }
        if (bRow < dims.K && col < dims.N) {
            tileB[lid.y][lid.x] = B[bRow * dims.N + col];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + tileA[lid.y][k] * tileB[k][lid.x];
        }
        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
"#;

/// NVIDIA CUDA C++ shared-memory tiled `f32` GEMM kernel (NVRTC compatible).
pub const CUDA_GEMM_F32_KERNEL: &str = r#"
extern "C" __global__ void gemm_f32_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
"#;
