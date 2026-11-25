# ADR-001: Language Selection and Modular Workspace Architecture

- **Status**: Accepted
- **Date**: 2026-11-25

## Context

We are building a long-term, independent AI computing infrastructure that must support tensor computation, computational graphs, automatic differentiation, training, inference, and future multi-backend hardware acceleration (Metal, CUDA, WebGPU, WebAssembly), while treating external model files and metadata as untrusted input.

## Decision

1. **Core Engine Language**: Stable **Rust** (Edition 2024).
   - Provides compile-time memory safety and data-race freedom across tensor storage, view lifetimes, and future concurrent graph execution.
   - Enables a single core engine to target Native binaries, Node.js (N-API), WebAssembly, and optional Python bindings (PyO3).
2. **Future Hardware Kernels**: Dedicated device kernel languages (SIMD intrinsics, MSL, CUDA/PTX, WGSL) isolated behind backend trait contracts.
3. **Strict Acyclic Workspace Layering**:
   - `as-core`: DTypes (`F32`, `F64`, `F16`, `BF16`, `I32`, `I64`), `Shape`, `Layout`, `Device`, `Error`, `MemoryLimits`. Zero dependencies.
   - `as-backend`: Hardware-independent `Backend` and `Storage` traits. Depends only on `as-core`.
   - `as-cpu`: Safe, deterministic reference CPU backend. Depends on `as-core` and `as-backend`.
   - `as-tensor`: Public ergonomic `Tensor<B>` API.
   - `as-bench`: Reproducible zero-dependency benchmark runner.

## Consequences

- Zero third-party compute framework dependencies (`ndarray`, `openblas`, `candle`, etc. are strictly excluded).
- Clean separation allows Phase 2 (Graph/Autograd) and Phase 7 (GPU Backends) to plug into `as-backend` without modifying `as-core` or `as-tensor` contracts.
