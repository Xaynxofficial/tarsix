<div align="center">

# TARSIX

**Independent, Zero-Dependency, Memory-Safe AI Computing Infrastructure**

[![CI](https://github.com/Xaynxofficial/tarsix/actions/workflows/ci.yml/badge.svg)](https://github.com/Xaynxofficial/tarsix/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/tarsix?style=flat-square&color=111827&label=npm%20tarsix)](https://www.npmjs.com/package/tarsix)
[![Rust](https://img.shields.io/badge/Rust-Edition%202024-111827?style=flat-square)](Cargo.toml)
[![Safe Rust](https://img.shields.io/badge/core-100%25%20Safe%20Rust-0f766e?style=flat-square)](SECURITY.md)
[![Dependencies](https://img.shields.io/badge/core%20dependencies-0-0f766e?style=flat-square)](Cargo.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-111827?style=flat-square)](LICENSE)

*Written from scratch in Rust — N-D Tensors, Memory Engine, Graph Compiler, Hardware Scheduler, Autograd, Training Pipeline, Model Hub, Model Profiler, Inference Server, Distributed Compute, Multi-Architecture LLMs, Computer Vision, Apple Silicon Metal GPU, 114 KiB WebAssembly, Python & C/C++ ABI.*

[Architecture](#architecture) · [Core Capabilities](#core-capabilities) · [Security Model](#security--reliability) · [Quick Start](#quick-start) · [Benchmarks](#benchmarks) · [Crates](#workspace-crates)

</div>

---

## Overview

**Tarsix** is a self-contained AI computation, training, and inference ecosystem built without wrapping PyTorch, TensorFlow, ONNX Runtime, BLAS, or third-party tensor crates.

Modern AI stacks routinely carry gigabytes of Python runtimes, hundreds of transitive dependencies, and C/C++ memory-safety liabilities when parsing untrusted weights. Tarsix is engineered around four strict principles:

1. **Zero-Dependency Core** — All 21 core Rust crates compile with zero external dependencies, eliminating transitive supply-chain risk.
2. **Memory-Safe & Panic-Free** — Workspace-wide `#![forbid(unsafe_code)]` in core crates, Clippy-enforced denial of `unwrap`, `expect`, and `panic`, and release-mode integer overflow checks (`overflow-checks = true`).
3. **Untrusted-Input Hardening** — External model weights, graphs, CSVs, and images are validated against explicit `MemoryLimits`, rank ceilings, recursion caps, and cryptographic-style integrity checksums before allocation.
4. **Single Core, Universal Targets** — One Rust computation foundation powering native Rust services, Apple Silicon Metal GPUs, a **40 kB gzipped (114 KiB raw)** WebAssembly npm package (`tarsix`), Python (`ctypes`), and C/C++ (`tarsix.h`).

---

## Architecture

Tarsix enforces a strict acyclic workspace hierarchy separating mathematical semantics, graph compilation, runtime execution, and hardware dispatch:

```mermaid
flowchart TB
    subgraph Interfaces["Language, Serving & Developer Interfaces"]
        CLI["as-cli"]
        SRV["as-serve (HTTP/1.1 + SSE Inference Server / Dynamic Batching)"]
        HUB["as-hub (Model Hub / SHA-256 Cache / Sharded Safetensors)"]
        TS["npm: tarsix (WASM / TypeScript)"]
        PY["Python (ctypes) & C/C++ (tarsix.h)"]
    end

    subgraph Applications["Domain Engines, Training & Distributed"]
        LLM["as-llm (Llama / Qwen2 / SmolLM / Mistral)"]
        VIS["as-vision (ViT / CNN / YOLO)"]
        DIST["as-dist (Ring-AllReduce / TCP Cluster / DDP)"]
        TRN["as-train & as-optim (Trainer Engine / Callbacks / ASCK / ASTS)"]
        RT["as-runtime & as-io (Profiler / Chrome Trace / SVG / Safetensors)"]
    end

    subgraph Compute["Scheduling, Differentiation, Compiler & Layers"]
        SCHED["as-scheduler (CPU/GPU Cost-Model Placement / DAG Waves)"]
        NN["as-nn (Layers / LoRA / Attention / Transformer)"]
        AG["as-autograd (Reverse Tape / JVP / Checkpoint)"]
        GR["as-graph (Graph Compiler: O0-O3 / Folding / CSE / FusedLinear / ASGR)"]
    end

    subgraph Foundation["Tensor, Memory & Hardware Abstraction"]
        TEN["as-tensor (Public N-D Tensor API)"]
        MEM["as-memory (Memory Engine: Budget Tracker / BufferPool / Static Planner)"]
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
    Compute --> MEM
    TEN --> BE
    BE --> CORE
    MEM --> CORE
    BE --> Hardware
```

---

## Core Capabilities

| Subsystem | Key Engineering Features |
| :--- | :--- |
| **Tensor & Memory Engine** | N-dimensional tensors, zero-copy `Shape`/`Stride`/`Offset` view algebra, independent IEEE-754 `F16` & `BF16` (exhaustively verified across all 65,536 bit patterns), plus **`as-memory`**: lock-free global `MemoryTracker` + RAII `MemoryLease`, power-of-two `BufferPool`/`TensorBufferPool`, and interval-coloring **Static Graph Memory Planner** (`plan_memory_reuse`). |
| **Graph Compiler** | Modular `PassPipeline` (`OptLevel::O0..=O3`) with **Constant Folding**, **Algebraic Simplification**, **Common Subexpression Elimination (CSE)**, **Dead-Code Elimination (DCE)**, **Unary Chain Fusion**, **Linear+Bias+Activation Fusion (`Op::FusedLinear`)**, static shape/dtype inference, and checksummed `ASGR` binary serialization. |
| **Hardware Scheduler** | Dynamic host CPU + live Apple Silicon Metal GPU discovery (`DeviceRegistry`), **Roofline Cost-Model Op Placement** (`CpuOnly`, `PreferGpu`, `CostModelAuto`), **ASAP Parallel DAG Wave Scheduler**, and `std::thread::scope` multi-branch concurrent execution (`as-scheduler`). |
| **Model Profiler & Runtime** | Op-level execution profiling (`OpProfile`), FLOPs & memory bandwidth (GB/s), **Roofline Bottleneck Classification** (`ComputeBound`/`MemoryBound`/`Overhead`), peak live memory tracking, **`chrome://tracing` / Perfetto JSON export**, **SVG Flamegraph timelines**, and CLI/Markdown summary tables (`as-runtime`). |
| **Automatic Differentiation** | Panic-free reverse-mode `Tape` with broadcast unbroadcasting, differentiable `im2col`/`col2im`, multi-step gradient accumulation, just-in-time recomputing **gradient checkpointing**, forward-mode `DualTensor` (JVP), and central finite-difference `gradcheck`. |
| **Training Pipeline & NN** | Declarative **`Trainer` engine** (`as-train`) with microbatch **gradient accumulation**, **loss scaling**, global L2 clipping, pluggable **`Callback` hooks** (`EarlyStopping`, `BestCheckpointCallback`, `JsonlLoggerCallback`), checksummed full-resume **`TrainState` (`ASTS`)**, `LoRALinear`, `TransformerBlock`, `AdamW`, `CosineAnnealingLr`, CSV I/O, and SVG charts. |
| **Model Hub & Interop** | Zero-dependency **`as-hub`** with pure safe Rust **FIPS 180-4 SHA-256**, content-addressable `ModelCache`, searchable `ModelCard`/`ModelQuery` registry, **Sharded Safetensors (`model.safetensors.index.json`)** loader with path-traversal blocking, and bounded JSON/Safetensors parser (`as-io`). |
| **Inference Server** | Zero-dependency hardened **HTTP/1.1 + SSE** server (`as-serve`) with real concurrent **`DynamicBatcher`**, vLLM-style **`ContinuousBatcher`** (iteration-level token scheduling), OpenAI-compatible `/v1/chat/completions` (JSON & SSE streaming), `/v1/infer`, `/v1/models`, `/health`, and **Prometheus `/metrics`**. |
| **Distributed Compute** | Zero-dependency **`as-dist`** with bandwidth-optimal **Ring-AllReduce** (`Sum`/`Mean`/`Max`/`Min`), `Broadcast`, `AllGather`, `Barrier`, local `LocalCluster`, multi-machine **`TcpCommunicator`** (`ASDC` checksummed framing), **`DistributedDataParallel` (DDP)**, and Megatron-LM **Tensor Parallelism** (`ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelMlp`). |
| **LLM & Vision Engines** | Multi-architecture HuggingFace LLM loader (`Llama-2/3`, `Qwen2/2.5`, `SmolLM/SmolLM2`, `Mistral`), GQA, RoPE, **`PagedKvCache` (PagedAttention + CoW prefix sharing)**, BPE tokenizer, **Int8 & 4-bit `Q4Linear`**, plus **ViT**, **CNN**, and trainable **YOLO `ObjectDetector`** (`BBox`, `IoU`, `NMS`). |
| **Hardware Acceleration** | Multi-core `std::thread::scope` cache-blocked CPU GEMM (100% safe Rust), Apple Silicon **`as-metal`** (Zero-Copy Unified Memory `MetalBuffer`, `MetalCommandBatch`, Tiled GEMM, RMSNorm, Softmax, SwiGLU, RoPE, Q4 MatVec), dynamic **`as-cuda`** (Linux NVIDIA CUDA / `cuBLAS` `dlopen` driver + LLM CUDA kernels), and differential consistency testing (`as-accel`). |

---

## Security & Reliability

Tarsix treats every external model file, serialized graph, dataset, and shape header as **untrusted input** (see [SECURITY.md](SECURITY.md)):

| Threat Class | Mitigation Enforced |
| :--- | :--- |
| Memory corruption & data races | `[workspace.lints.rust] unsafe_code = "forbid"` inherited via `[lints] workspace = true` across all 21 core crates; isolated audited FFI only in `as-metal`, `as-cuda`, & `as-wasm` (which omit workspace lint inheritance; see [SECURITY.md](SECURITY.md)). |
| Runtime panics / DoS crashes | Workspace Clippy denies `unwrap_used`, `expect_used`, `panic`; all fallible APIs return `Result<T>`. |
| Integer overflow in shape/stride math | `overflow-checks = true` in release profile + `checked_mul`/`checked_add` everywhere. |
| OOM allocation bombs | Pre-allocation `MemoryLimits` (`max_tensor_bytes`, `MAX_RANK = 16`, `MAX_IMAGE_DIM = 16_384`). |
| Crafted Safetensors / JSON payloads | Header byte cap (`16 MiB`), `MAX_JSON_DEPTH = 32`, overlapping `data_offsets` rejection, path-traversal blocking on shards. |
| Corrupted or tampered weights/graphs | Mandatory FNV-1a 64-bit integrity checksums on `ASGR` graphs and `ASCK` checkpoints (no unsafe pickle/RCE formats). |

---

## Quick Start & Usage Guides

### 1. Using Tarsix in Rust Code (`Tensor` & `Autograd`)

Add the crates you need from this workspace (or git URL) to your `Cargo.toml`:

```toml
[dependencies]
as-tensor   = { git = "https://github.com/Xaynxofficial/tarsix" }
as-autograd = { git = "https://github.com/Xaynxofficial/tarsix" }
```

```rust
use as_autograd::Tape;
use as_tensor::{DType, Result, Tensor};

fn main() -> Result<()> {
    // 1. Tensor creation, broadcasting & multi-core tiled MatMul
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    let bias = Tensor::from_f32(&[1.0, -2.0], vec![1, 2])?;
    let out = a.matmul(&b)?.add(&bias)?;
    println!("MatMul + bias = {:?}", out.to_f32_vec()?); // [20.0, 20.0, 44.0, 48.0]

    // 2. Automatic Differentiation (Reverse-Mode Tape)
    let mut tape = Tape::new();
    let x = tape.var(Tensor::from_f64(&[2.0, 3.0], vec![1, 2])?, true);
    let w = tape.var(Tensor::from_f64(&[0.5, -1.0], vec![2, 1])?, true);
    let y = tape.matmul(x, w)?;
    let loss = tape.powf(y, 2.0)?;
    let grads = tape.backward(loss)?;
    println!("dL/dw = {:?}", grads.get(w).unwrap().to_f64_vec()?);
    Ok(())
}
```

### 2. Training & Predicting from the Terminal (`as-cli`)

Clone the repo and run commands directly — no Python environment required:

```bash
git clone https://github.com/Xaynxofficial/tarsix.git
cd tarsix

# Verify all 52 unit, numerical, compiler, scheduler, server, distributed, and security tests
cargo test --workspace

# Show detected CPU/GPU hardware & security posture
cargo run --release -p as-cli -- info
```

**Prepare your own CSV dataset** (numeric feature columns, last column = integer class `0, 1, 2...`):
```csv
height_m,weight_100kg,label
1.80,0.52,0
1.75,0.68,1
1.70,0.92,2
```

**Train, save a checksummed `.asck` model, and run live predictions:**
```bash
# Or generate a ready-to-use 240-sample 3-class CSV dataset
cargo run --release -p as-cli -- make-dataset /tmp/data.csv 240

# Train (80% train / 20% validation split, AdamW + CosineAnnealingLr + grad clipping)
# Usage: train <data.csv> <out.asck> [epochs] [hidden_units]
cargo run --release -p as-cli -- train /tmp/data.csv /tmp/model.asck 80 24

# Predict on new unseen inputs
cargo run --release -p as-cli -- predict /tmp/model.asck "0.1,0.1"
```

**Train a mini Causal Transformer from scratch & chat:**
```bash
cargo run --release -p as-cli -- demo-chat $'S: merhaba\nC:'
```

### 3. LoRA Fine-Tuning, 4-Bit Quantization (`Q4Linear`) & YOLO Object Detection

**Parameter-Efficient LoRA Fine-Tuning & 4-Bit Compression:**
```rust
use as_llm::Q4Linear;
use as_nn::{Linear, LoRALinear, Rng};
use as_tensor::DType;

let mut rng = Rng::from_seed(42);
let base = Linear::new("proj", 512, 512, false, DType::F32, &mut rng)?;

// Wrap frozen base layer with rank-8 LoRA adapters (only ~3% trainable params)
let lora = LoRALinear::from_linear("proj", &base, 8, 16.0, &mut rng)?;
// After training: fold LoRA back into base weights for zero-overhead inference
let merged = lora.merge_weights()?;

// Compress weights to 4-bit blocks (~7.1x smaller than f32)
let q4 = Q4Linear::from_f32_weight(merged.weight.data())?;
```

**Trainable YOLO-Style Object Detection (`BBox`, `IoU`, `NMS`):**
```rust
use as_autograd::Tape;
use as_vision::{BBox, LabeledBox, ObjectDetector, decode_detections, detection_loss, encode_yolo_targets};

let detector = ObjectDetector::new(3, 16, 2, 42)?;
let gt = vec![vec![LabeledBox {
    bbox: BBox { cx: 0.62, cy: 0.37, w: 0.25, h: 0.25 },
    class_id: 1,
}]];
let target_grid = encode_yolo_targets(&gt, 4, 4, 2)?;
// Forward + backward on Tape with `detection_loss`, then decode with `decode_detections(..., 0.5, 0.5)`
```

### 4. Running Real HuggingFace LLMs (Llama / Qwen2 / SmolLM / Mistral)

```bash
# 1. Download a compact Safetensors model (e.g., SmolLM2-135M-Instruct, ~270 MB)
mkdir -p /tmp/smollm2 && cd /tmp/smollm2
curl -LO https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json
curl -LO https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json
curl -LO https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors

# 2. Run streaming autoregressive generation directly with Tarsix
cd /path/to/tarsix
cargo run --release -p as-cli -- run-llm /tmp/smollm2 "Artificial intelligence is" 40 0.7
```

### 5. TypeScript / JavaScript (`npm install tarsix`)

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

### 6. Python (`ctypes`) & C / C++ (`include/tarsix.h`)

Calls the native multi-core Rust library directly via C ABI with zero `pip` dependencies:

```bash
# Build native shared library (target/release/libas_wasm.dylib | .so | .dll)
cargo build -p as-wasm --release

# Run Python example
python3 sdk/python/tarsix.py
```

```python
from sdk.python.tarsix import Tensor

a = Tensor.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
b = Tensor.from_list([5.0, 6.0, 7.0, 8.0], [2, 2])
print((a @ b).to_list())  # [19.0, 22.0, 43.0, 50.0]
```

```c
#include "include/tarsix.h"

uint32_t shape[2] = {2, 2};
float a_vals[4] = {1, 2, 3, 4};
float b_vals[4] = {5, 6, 7, 8};
uint32_t a = as_tensor_from_f32(a_vals, 4, shape, 2);
uint32_t b = as_tensor_from_f32(b_vals, 4, shape, 2);
uint32_t c = as_tensor_matmul(a, b);
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
<summary><strong>Inspect all 24 crates & SDKs</strong></summary>

- `crates/as-core` — Core types, errors, `MemoryLimits`, `F16`/`BF16`, `Shape`, `Layout`
- `crates/as-memory` — Memory Engine: lock-free `MemoryTracker`, `BufferPool`, static interval-coloring `MemoryPlan`
- `crates/as-backend` — `Backend` and `Storage` hardware abstraction traits
- `crates/as-cpu` — Safe multi-core tiled CPU execution engine
- `crates/as-metal` — Apple Silicon `Metal.framework` GPU backend (Unified Memory, command batching, LLM MSL kernels)
- `crates/as-cuda` — Dynamic NVIDIA CUDA / cuBLAS `dlopen` FFI backend + LLM CUDA kernel sources
- `crates/as-tensor` — Public `Tensor<B>` Rust API
- `crates/as-graph` — Graph Compiler (`O0`–`O3`), Constant Folding, CSE, Algebraic Simplify, `FusedLinear`, shape inference, `ASGR` codec
- `crates/as-scheduler` — Hardware Scheduler: dynamic CPU/Metal discovery, cost-model placement, parallel DAG wave execution
- `crates/as-autograd` — Reverse-mode `Tape`, checkpointing, forward-mode `DualTensor`, `gradcheck`
- `crates/as-nn` — Neural network layers, `LoRALinear`, activations, losses, attention, Transformer
- `crates/as-optim` — `Sgd`, `Adam`, `AdamW`, LR schedulers, gradient clipping
- `crates/as-train` — Training Pipeline: declarative `Trainer`, `Callback`s (`EarlyStopping`, `BestCheckpoint`, JSONL), `TrainState` resume, `Dataset`, `DataLoader`, CSV, `ASCK`, SVG charts
- `crates/as-io` — Bounded JSON parser, Safetensors reader/writer, `ModelConfig`
- `crates/as-hub` — Model Hub: pure Rust SHA-256, content-addressable `ModelCache`, searchable `ModelCard` registry, sharded Safetensors loader
- `crates/as-runtime` — `InferenceSession`, op-level Model Profiler, Roofline analysis, Chrome Trace JSON & SVG flamegraphs
- `crates/as-serve` — Inference Server: hardened HTTP/1.1 + SSE, `DynamicBatcher`, OpenAI `/v1/chat/completions`, `/v1/infer`, Prometheus `/metrics`
- `crates/as-dist` — Distributed Compute: Ring-AllReduce, `LocalCluster`, `TcpCommunicator` (`ASDC` framing), `DistributedDataParallel` (DDP)
- `crates/as-llm` — Multi-architecture LLM decoder, GQA, RoPE, KV-Cache, BPE, Int8 & Q4 quantization
- `crates/as-vision` — Image preprocessing, `PatchEmbed`, `VisionTransformer`, `CnnClassifier`, YOLO `ObjectDetector`
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
