# Security Policy & Threat Model

## Security Posture

This project treats all external model files, serialized graphs, shapes, and user-provided tensor metadata as **untrusted input**.

### Compile-Time & Runtime Guarantees (Milestone 1)

1. **Zero `unsafe` in Core & Reference CPU Engine**: Workspace-wide `#![forbid(unsafe_code)]` eliminates memory corruption classes (buffer overflows, use-after-free, data races) across all core tensor, layout, storage, and CPU execution code.
2. **Panic-Free Production Paths**: Workspace Clippy lints deny `unwrap_used`, `expect_used`, and `panic` in library crates. All fallible operations return strongly typed `as_core::Result<T>`.
3. **Integer Overflow Protection**:
   - Release builds enable `overflow-checks = true`.
   - All shape element counts, stride offsets, and byte-size calculations use checked arithmetic (`checked_mul`, `checked_add`).
4. **Resource Exhaustion (OOM / DoS) Prevention**:
   - Explicit `MemoryLimits` enforce maximum tensor rank (`MAX_RANK`) and maximum single-allocation byte budgets before allocating heap memory.
5. **Zero Third-Party Supply-Chain Surface**:
   - Core crates have zero external crate dependencies, eliminating transitive supply-chain vulnerabilities.

### Future Unsafe Isolation Policy (Hardware Acceleration)

When SIMD intrinsics, Metal, CUDA, or WebGPU backends are introduced in Phase 7:
- `unsafe` blocks are strictly forbidden in `as-core`, `as-backend`, and `as-tensor`.
- Any `unsafe` kernel code must live in a dedicated, audited backend crate with explicit safety invariants, bounds validation at the safe boundary, and differential testing against the safe CPU reference backend.
