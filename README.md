<div align="center">

# TARSIX

**Independent, Zero-Dependency, Memory-Safe AI Computing Infrastructure**

[![CI](https://github.com/Xaynxofficial/tarsix/actions/workflows/ci.yml/badge.svg)](https://github.com/Xaynxofficial/tarsix/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/tarsix?style=flat-square&color=111827&label=npm%20tarsix)](https://www.npmjs.com/package/tarsix)
[![Rust](https://img.shields.io/badge/Rust-Edition%202024-111827?style=flat-square)](Cargo.toml)
[![Unsafe](https://img.shields.io/badge/unsafe__code-forbid%20(core)-0f766e?style=flat-square)](SECURITY.md)
[![Dependencies](https://img.shields.io/badge/core%20dependencies-0-0f766e?style=flat-square)](Cargo.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-111827?style=flat-square)](LICENSE)

*Written from scratch in Rust — N-D Tensors, Computational Graphs, Autograd, Neural Networks, Training, Safetensors, Multi-Architecture LLMs, Computer Vision, Apple Silicon Metal GPU, 114 KiB WebAssembly, Python & C/C++ ABI.*

[Architecture](#architecture) · [Core Capabilities](#core-capabilities) · [Security Model](#security--reliability) · [Quick Start](#quick-start) · [Benchmarks](#benchmarks) · [Crates](#workspace-crates)

</div>

---

## Overview

**Tarsix** is a self-contained AI computation, training, and inference ecosystem built without wrapping PyTorch, TensorFlow, ONNX Runtime, BLAS, or third-party tensor crates.

Modern AI stacks routinely carry gigabytes of Python runtimes, hundreds of transitive dependencies, and C/C++ memory-safety liabilities when parsing untrusted weights. Tarsix is engineered around four strict principles:

1. **Zero-Dependency Core** — All 16 core Rust crates compile with zero external dependencies, eliminating transitive supply-chain risk.
2. **Memory-Safe & Panic-Free** — Workspace-wide `#![forbid(unsafe_code)]` in core crates, Clippy-enforced denial of `unwrap`, `expect`, and `panic`, and release-mode integer overflow checks (`overflow-checks = true`).
3. **Untrusted-Input Hardening** — External model weights, graphs, CSVs, and images are validated against explicit `MemoryLimits`, rank ceilings, recursion caps, and cryptographic-style integrity checksums before allocation.
4. **Single Core, Universal Targets** — One Rust computation foundation powering native Rust services, Apple Silicon Metal GPUs, a **40 kB gzipped (114 KiB raw)** WebAssembly npm package (`tarsix`), Python (`ctypes`), and C/C++ (`tarsix.h`).

---

## Architecture

Tarsix enforces a strict acyclic workspace hierarchy separating mathematical semantics, graph compilation, runtime execution, and hardware dispatch:

```mermaid
flowchart TB
    subgraph Interfaces["Language & Developer Interfaces"]
        CLI["as-cli"]
        TS["npm: tarsix (WASM / TypeScript)"]
        PY["Python (ctypes) & C/C++ (tarsix.h)"]
    end

    subgraph Applications["Domain Engines & Runtime"]
        LLM["as-llm (Llama / Qwen2 / SmolLM / Mistral)"]
        VIS["as-vision (ViT / CNN / Preprocess)"]
        TRN["as-train & as-optim (AdamW / CSV / ASCK)"]
        RT["as-runtime & as-io (Safetensors / Profiling)"]
    end

    subgraph Compute["Differentiation, Graphs & Layers"]
        NN["as-nn (Layers / Attention / Transformer)"]
        AG["as-autograd (Reverse Tape / JVP / Checkpoint)"]
        GR["as-graph (DAG IR / DCE / Unary Fusion / ASGR)"]
    end

    subgraph Foundation["Tensor & Hardware Abstraction"]
        TEN["as-tensor (Public N-D Tensor API)"]
        BE["as-backend (Backend & Storage Traits)"]
        CORE["as-core (DType / F16 / BF16 / Shape / Limits)"]
    end

    subgraph Hardware["Execution Backends"]
        CPU["as-cpu (Multi-Core Tiled Safe Rust)"]
        MTL["as-metal (Apple Silicon Metal.framework)"]
        ACC["as-accel (MSL / WGSL / CUDA Shaders & Consistency)"]
    end

    Interfaces --> Applications
    Applications --> Compute
    Compute --> TEN
    TEN --> BE
    BE --> CORE
    BE --> Hardware
```

---

## Core Capabilities

| Subsystem | Key Engineering Features |
| :--- | :--- |
| **Tensor Engine** | N-dimensional tensors, zero-copy `Shape`/`Stride`/`Offset` view algebra (`transpose`, `permute`, `slice`, `reshape`, `broadcast_to`), independent IEEE-754 `F16` & `BF16` (round-to-nearest-even, exhaustively verified across all 65,536 bit patterns), `F32`, `F64`, `I32`, `I64`. |
| **Computational Graph** | Iterative Kahn topological scheduler with cycle detection, Dead-Code Elimination (DCE), single-pass Unary Operation Fusion (`FusedUnaryChain`), liveness-based intermediate buffer release, checksummed `ASGR` binary serialization. |
| **Automatic Differentiation** | Panic-free reverse-mode `Tape` with broadcast unbroadcasting, differentiable `im2col`/`col2im`, multi-step gradient accumulation, just-in-time recomputing **gradient checkpointing**, forward-mode `DualTensor` (JVP), and central finite-difference `gradcheck`. |
| **Neural Networks & Training** | Deterministic `Xoshiro256++` PRNG, `Linear`, `Embedding`, `LayerNorm`, `RmsNorm`, `Dropout`, `Conv2d`, `MultiHeadAttention`, `TransformerBlock`, `Sgd`/`Adam`/`AdamW`, `StepLr`/`CosineAnnealingLr`, global L2 gradient clipping, CSV dataset I/O, and checksummed `ASCK` checkpoints. |
| **Model Interoperability & Runtime** | Recursion-bounded JSON parser, zero-dependency **Safetensors** reader/writer (`F32/F64/F16/BF16/I32/I64`), `InferenceSession` with graph optimization, latency profiling, dynamic request batching, and streaming step execution. |
| **LLM Engine** | Multi-architecture HuggingFace loader (`Llama-2/3`, `Qwen2/2.5`, `SmolLM/SmolLM2`, `Mistral`, `TinyLlama`), Grouped-Query Attention (GQA), RoPE, bounded `KvCache`, HuggingFace `tokenizer.json` BPE & `ByteTokenizer`, Int8 per-channel `QuantizedLinear`, and Temperature/Top-k/Top-p streaming generation. |
| **Computer Vision** | Validated `ImageBuffer`, bilinear resizing, ImageNet CHW normalization, `PatchEmbed`, bidirectional `VisionTransformer` (ViT), `max_pool2d`, and end-to-end trainable `CnnClassifier`. |
| **Hardware Acceleration** | Multi-core `std::thread::scope` cache-blocked CPU GEMM (100% safe Rust), isolated Apple Silicon `Metal.framework` GPU backend (`as-metal`), versioned MSL/WGSL/CUDA kernel sources, and differential cross-backend numerical consistency testing (`as-accel`). |

---

## Security & Reliability

Tarsix treats every external model file, serialized graph, dataset, and shape header as **untrusted input** (see [SECURITY.md](SECURITY.md)):

| Threat Class | Mitigation Enforced |
| :--- | :--- |
| Memory corruption & data races | `#![forbid(unsafe_code)]` across all 16 core crates; narrow audited FFI only in `as-metal` & `as-wasm`. |
| Runtime panics / DoS crashes | Workspace Clippy denies `unwrap_used`, `expect_used`, `panic`; all fallible APIs return `Result<T>`. |
| Integer overflow in shape/stride math | `overflow-checks = true` in release profile + `checked_mul`/`checked_add` everywhere. |
| OOM allocation bombs | Pre-allocation `MemoryLimits` (`max_tensor_bytes`, `MAX_RANK = 16`, `MAX_IMAGE_DIM = 16_384`). |
| Crafted Safetensors / JSON payloads | Header byte cap (`16 MiB`), `MAX_JSON_DEPTH = 32`, overlapping `data_offsets` rejection, path-traversal blocking on shards. |
| Corrupted or tampered weights/graphs | Mandatory FNV-1a 64-bit integrity checksums on `ASGR` graphs and `ASCK` checkpoints (no unsafe pickle/RCE formats). |

---

## Quick Start

### 1. Rust Workspace & CLI

```bash
# Run the full verification suite (unit, numerical, gradcheck, JVP==VJP, security)
cargo test --workspace

# Run reproducible CPU benchmarks
cargo run --release -p as-bench

# Hardware & security diagnostics
cargo run --release -p as-cli -- info
```

**Train & predict on any CSV dataset:**
```bash
cargo run --release -p as-cli -- make-dataset /tmp/data.csv 240
cargo run --release -p as-cli -- train /tmp/data.csv /tmp/model.asck 80 24
cargo run --release -p as-cli -- predict /tmp/model.asck "0.1,0.1"
```

**Run any compatible HuggingFace Safetensors LLM (SmolLM2, Qwen2.5, Llama, Mistral):**
```bash
cargo run --release -p as-cli -- run-llm /path/to/hf_model_dir "Artificial intelligence is" 40 0.7
```

### 2. TypeScript / JavaScript (`npm install tarsix`)

Works in **Node.js (>=18)** and **Web Browsers** with the **114 KiB** WebAssembly engine bundled — zero external dependencies, no Rust or Python required:

```bash
npm install tarsix
```

```ts
import { createEngine, Tensor } from "tarsix";

const engine = await createEngine();
const a = Tensor.fromF32(engine, [1, 2, 3, 4], [2, 2]);
const b = Tensor.fromF32(engine, [5, 6, 7, 8], [2, 2]);
const bias = Tensor.fromF32(engine, [1, -2], [1, 2]);

const out = a.matmul(b).add(bias);
console.log(out.toF32Array()); // Float32Array(4) [ 20, 20, 44, 48 ]

a.dispose(); b.dispose(); bias.dispose(); out.dispose();
```

### 3. Python (`ctypes`) & C / C++ (`include/tarsix.h`)

Calls the native multi-core Rust library directly via C ABI with zero `pip` dependencies:

```bash
cargo build -p as-wasm --release
python3 sdk/python/tarsix.py
```

```python
from sdk.python.tarsix import Tensor

a = Tensor.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
b = Tensor.from_list([5.0, 6.0, 7.0, 8.0], [2, 2])
print((a @ b).to_list())  # [19.0, 22.0, 43.0, 50.0]
```

---

## Benchmarks

Reproducible measurements collected via `cargo run --release -p as-bench` on **Apple Silicon (`aarch64` / macOS, 8 threads, 100% safe Rust, zero external BLAS)**:

| Workload | Median (ms) | Mean (ms) | P95 (ms) | Throughput |
| :--- | ---: | ---: | ---: | ---: |
| `add_f32_contiguous_1M` | 0.235 | 0.207 | 0.240 | **50.98 GB/s** |
| `add_f32_broadcast_1024x1024` | 0.149 | 0.150 | 0.196 | **56.38 GB/s** |
| `matmul_f32_256x256` | 0.492 | 0.528 | 0.800 | **68.15 GFLOP/s** |
| `matmul_f32_512x512` | 3.870 | 3.831 | 5.203 | **69.37 GFLOP/s** |

*All performance claims are strictly tied to reproducible workloads in [`crates/as-bench`](crates/as-bench).*

---

## Workspace Crates

<details>
<summary><strong>Inspect all 18 crates & SDKs</strong></summary>

- `crates/as-core` — Core types, errors, `MemoryLimits`, `F16`/`BF16`, `Shape`, `Layout`
- `crates/as-backend` — `Backend` and `Storage` hardware abstraction traits
- `crates/as-cpu` — Safe multi-core tiled CPU execution engine
- `crates/as-metal` — Apple Silicon `Metal.framework` GPU backend
- `crates/as-tensor` — Public `Tensor<B>` Rust API
- `crates/as-graph` — Computational graph IR, DCE, Unary Fusion, `ASGR` codec
- `crates/as-autograd` — Reverse-mode `Tape`, checkpointing, forward-mode `DualTensor`, `gradcheck`
- `crates/as-nn` — Neural network layers, activations, losses, attention, Transformer
- `crates/as-optim` — `Sgd`, `Adam`, `AdamW`, LR schedulers, gradient clipping
- `crates/as-train` — `Dataset`, `DataLoader`, CSV I/O, metrics, `ASCK` checkpoints
- `crates/as-io` — Bounded JSON parser, Safetensors reader/writer, `ModelConfig`
- `crates/as-runtime` — `InferenceSession`, profiling, dynamic batching, streaming
- `crates/as-llm` — Multi-architecture LLM decoder, GQA, RoPE, KV-Cache, BPE, Int8 quantization
- `crates/as-vision` — Image preprocessing, `PatchEmbed`, `VisionTransformer`, `CnnClassifier`
- `crates/as-accel` — Hardware discovery, MSL/WGSL/CUDA kernels, differential consistency harness
- `crates/as-wasm` — Zero-dependency WebAssembly & native C-ABI `cdylib`
- `crates/as-cli` — Command-line interface (`as-cli`)
- `crates/as-bench` — Zero-dependency statistical benchmark runner
- `sdk/typescript` — Published `tarsix` npm package
- `sdk/python` & `include/tarsix.h` — Python `ctypes` & C/C++ bindings

</details>

---

## Architecture Decision Records

- [ADR-001: Language Selection and Modular Workspace Architecture](docs/adr/ADR-001-language-and-workspace-architecture.md)
- [ADR-002: Memory Safety, Panic-Free APIs, and Untrusted-Input Security Model](docs/adr/ADR-002-memory-safety-and-security-model.md)
- [ADR-003: Tensor Layout, Zero-Copy Views, and Independent Half-Precision Types](docs/adr/ADR-003-tensor-layout-and-dtype-system.md)
- [ADR-004: Computational Graph IR, Operation Fusion, and Panic-Free Autodiff Tape](docs/adr/ADR-004-graph-and-autograd-architecture.md)

---

## Contributing & License

Contributions are welcome — please review [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before opening a pull request.

Licensed under the **Apache License, Version 2.0** ([LICENSE](LICENSE)).
