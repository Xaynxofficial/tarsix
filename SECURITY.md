# Security Policy & Threat Model

## Security Posture

Tarsix treats every external model file, serialized graph, checkpoint, dataset, image, HTTP request, and distributed network frame as **untrusted input**.

---

## Compile-Time & Runtime Guarantees

1. **Zero `unsafe` Across All 21 Core Crates**:
   - The root [Cargo.toml](Cargo.toml) defines `[workspace.lints.rust] unsafe_code = "forbid"`.
   - Every core engine crate (`as-core`, `as-memory`, `as-backend`, `as-cpu`, `as-tensor`, `as-graph`, `as-scheduler`, `as-autograd`, `as-nn`, `as-optim`, `as-train`, `as-io`, `as-hub`, `as-runtime`, `as-serve`, `as-dist`, `as-llm`, `as-vision`, `as-accel`, `as-cli`, `as-bench`) explicitly opts into these lints via `[lints] workspace = true`.
   - This eliminates memory corruption classes (buffer overflows, use-after-free, data races) across all tensor math, memory management, graph compilation, scheduling, training, hub caching, HTTP serving, and distributed networking code.
2. **Panic-Free Production Paths**:
   - Workspace Clippy lints deny `unwrap_used`, `expect_used`, and `panic` in library crates. All fallible operations return strongly typed `as_core::Result<T>`.
3. **Integer Overflow Protection**:
   - Release builds enable `overflow-checks = true`.
   - Shape element counts, stride offsets, buffer sizes, and frame lengths use checked arithmetic (`checked_mul`, `checked_add`).
4. **Resource Exhaustion (OOM / DoS) Prevention**:
   - Pre-allocation `MemoryLimits` (`max_tensor_bytes`, `MAX_RANK = 16`, `MAX_IMAGE_DIM = 16_384`) and `as-memory::MemoryBudget` enforce strict allocation ceilings.
   - `as-io` caps Safetensors headers (`16 MiB`), JSON recursion depth (`MAX_JSON_DEPTH = 32`), and rejects overlapping `data_offsets`.
   - `as-serve` enforces `max_header_bytes`, `max_body_bytes`, `read_timeout_ms`, and `max_concurrent_requests` (`429` / `413` protection).
   - `as-dist` enforces `max_frame_bytes` on all `ASDC` TCP collective frames.
5. **Cryptographic & Checksum Integrity + Path-Traversal Hardening**:
   - `as-hub` verifies all cached model artifacts using pure safe Rust **FIPS 180-4 SHA-256** and rejects path traversal (`..`, `/`, `\`) in sharded `model.safetensors.index.json` filenames.
   - `ASGR` graphs, `ASCK` checkpoints, `ASTS` training states, and `ASDC` distributed TCP frames are protected by mandatory FNV-1a 64-bit integrity checksums (no unsafe pickle/RCE formats).
6. **Zero Third-Party Supply-Chain Surface**:
   - All core crates compile with zero external crate dependencies, eliminating transitive supply-chain vulnerabilities.

---

## FFI & `unsafe` Isolation Policy (`as-metal`, `as-cuda`, and `as-wasm`)

In Cargo, `[workspace.lints]` is **opt-in per crate** via `[lints] workspace = true`. This allows Tarsix to enforce `unsafe_code = "forbid"` strictly across all 21 core crates while isolating unavoidable platform FFI into exactly **three dedicated boundary crates** that intentionally omit `[lints] workspace = true`:

| Boundary Crate | Purpose of Isolated `unsafe` | Mandatory Safeguards |
| :--- | :--- | :--- |
| **`crates/as-metal`** | Narrow Objective-C / C-ABI bridge to Apple's `Metal.framework` GPU driver on macOS (`aarch64`), including Unified Memory `MetalBuffer` and batched command encoding. | All shapes, dtypes, strides, and `MemoryLimits` are validated in safe Rust before crossing FFI; every `unsafe` block documents explicit `// SAFETY:` invariants; outputs are differentially tested against the safe `as-cpu` reference backend. |
| **`crates/as-cuda`** | Runtime `dlopen` C-ABI bridge to `libcudart.so` and `libcublas.so` on Linux NVIDIA GPU hosts (zero build-time CUDA Toolkit dependency). | All shapes, lengths, and `MemoryLimits` are checked before device allocation; device buffers are unconditionally freed on all error paths; clean `Result::Err` fallback on non-CUDA hosts. |
| **`crates/as-wasm`** | C-ABI (`extern "C"`) and WebAssembly (`wasm32-unknown-unknown`) `cdylib` exports for TypeScript, Python (`ctypes`), and C/C++ (`include/tarsix.h`). | Raw pointers and lengths from foreign callers are null-checked and bounds-checked at the FFI entry point before constructing slices; internal tensor state is referenced via opaque integer handles rather than exposing raw Rust pointers. |

### Rules for Any Future Hardware Backend (e.g. CUDA, WebGPU, SIMD)
- `unsafe` remains strictly forbidden in all 21 core crates (never weaken `[workspace.lints.rust] unsafe_code = "forbid"`).
- Any new native driver or hardware kernel requiring `unsafe` must live in its own isolated driver crate, validate all inputs at the safe Rust boundary, document `// SAFETY:` invariants, and pass differential numerical consistency tests against `as-cpu`.
