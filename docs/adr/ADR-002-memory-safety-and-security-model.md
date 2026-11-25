# ADR-002: Memory Safety, Panic-Free APIs, and Untrusted-Input Security Model

- **Status**: Accepted
- **Date**: 2026-11-25

## Context

AI runtimes routinely load untrusted external weights, shapes, and serialized graphs. Traditional C/C++ frameworks suffer from heap buffer overflows, integer overflows in shape/stride arithmetic, and OOM denial-of-service attacks.

## Decision

1. **Workspace-Wide `#![forbid(unsafe_code)]`**: Enabled in root `[workspace.lints.rust]` for Milestone 1. Every crate in the core engine is 100% safe Rust.
2. **Panic-Free Production Code**: Workspace Clippy lints set `unwrap_used = "deny"`, `expect_used = "deny"`, and `panic = "deny"`. All fallible operations return `as_core::Result<T>`.
3. **Checked Arithmetic Everywhere**:
   - `overflow-checks = true` in release profile.
   - `Shape::new`, `Layout::physical_index`, `Layout::slice`, and `MemoryLimits::check_allocation` use `checked_mul` and `checked_add`.
   - Integer tensor operations (`I32`, `I64`) return `Error::ArithmeticOverflow` on overflow and `Error::InvalidArgument` on division by zero.
4. **Resource Budgets**: `MemoryLimits` enforces `MAX_RANK = 16` and configurable `max_tensor_bytes` before heap allocation occurs.
5. **Future Unsafe Isolation (Phase 7)**: When SIMD/GPU backends require `unsafe`, it will be confined exclusively to audited hardware backend crates and differentially tested against `as-cpu`.
