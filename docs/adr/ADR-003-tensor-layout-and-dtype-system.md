# ADR-003: Tensor Layout, Zero-Copy Views, and Independent Half-Precision Types

- **Status**: Accepted
- **Date**: 2026-11-25

## Context

High-performance tensor engines avoid unnecessary memory copies during `transpose`, `permute`, `reshape`, `slice`, and `broadcast_to`, while supporting modern ML numeric types (`F16`, `BF16`) without relying on external crates.

## Decision

1. **Zero-Copy View Algebra**:
   - `Tensor` pairs an `Arc`-shared `Storage` handle with a `Layout { shape, strides, offset }`.
   - `broadcast_to` uses `0` strides for expanded dimensions.
   - `transpose`, `permute`, `slice`, `squeeze`, and `unsqueeze` manipulate only `Layout` metadata.
2. **Independent `F16` and `BF16`**:
   - Implemented directly in `as-core::dtype` with IEEE-754 round-to-nearest-even semantics, subnormal support, and NaN/Inf preservation.
   - Verified exhaustively across all 65,536 possible 16-bit patterns.
   - `F16`/`BF16` matrix multiplication and reductions accumulate in `f32` to avoid catastrophic precision loss before rounding back.
