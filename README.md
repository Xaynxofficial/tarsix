# Tarsix — Independent AI Computing Infrastructure

> Zero-dependency, `#![forbid(unsafe_code)]`, security-hardened AI computing engine built from scratch in **Rust** (Edition 2024), with **Apple Silicon Metal GPU**, **114 KiB WebAssembly (npm `tarsix`)**, **Python**, and **C/C++ ABI** support.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](Cargo.toml)
[![npm](https://img.shields.io/badge/npm-tarsix-red.svg)](https://www.npmjs.com/package/tarsix)

---

## Highlights

- **Zero External Runtime Dependencies** in all core Rust crates — eliminating supply-chain attack surface.
- **Memory-Safe & Panic-Free Core**: `#![forbid(unsafe_code)]` across all 16 core crates, `unwrap_used`/`expect_used`/`panic` denied via workspace Clippy policy, release-mode `overflow-checks = true`.
- **Untrusted-Input Security**: `MemoryLimits` OOM guards, `MAX_RANK = 16`, recursion-bounded JSON parser, Safetensors offset-overlap rejection, and FNV-1a 64-bit integrity checksums on graphs (`ASGR`) and checkpoints (`ASCK`).
- **Multi-Language Reach**: Native **Rust**, **TypeScript/JavaScript (`npm install tarsix`, 40 kB gzipped WASM)**, **Python (`ctypes`)**, and **C/C++ (`include/tarsix.h`)**.
- **Full AI Stack**: N-D Tensors (`F32/F64/F16/BF16/I32/I64`), Graph IR & Fusion, Reverse & Forward Autodiff + Checkpointing, Neural Networks, Optimizers (`SGD/Adam/AdamW`), CSV Training Pipeline, Safetensors & Inference Runtime, Multi-Architecture LLM Engine (`Llama/Qwen2/Mistral/SmolLM` with GQA, RoPE, KV-Cache, BPE, Int8), Vision (`ViT` & `CNN`), and Apple Silicon Metal GPU backend.

---

## Workspace Architecture (18 Crates)

| Crate / Path | Purpose |
| :--- | :--- |
| [`crates/as-core`](crates/as-core) | `Error`, `MemoryLimits`, `Device`, independent IEEE-754 `F16`/`BF16`, `Shape`, `Layout`, zero-copy view algebra |
| [`crates/as-backend`](crates/as-backend) | Hardware-independent `Backend` & `Storage` trait contracts |
| [`crates/as-cpu`](crates/as-cpu) | Safe, deterministic, multi-core parallel (`std::thread::scope`) tiled CPU backend |
| [`crates/as-metal`](crates/as-metal) | Isolated Apple Silicon `Metal.framework` GPU backend (`gemm_f32_tiled`, `add_f32`) |
| [`crates/as-tensor`](crates/as-tensor) | Ergonomic, backend-generic public Rust `Tensor<B>` API |
| [`crates/as-graph`](crates/as-graph) | DAG IR, cycle-checked scheduler, DCE & Unary Fusion optimizer, checksummed `ASGR` codec |
| [`crates/as-autograd`](crates/as-autograd) | Panic-free reverse-mode `Tape`, `im2col`/`col2im`, recomputing checkpointing, forward-mode `DualTensor`, `gradcheck` |
| [`crates/as-nn`](crates/as-nn) | Deterministic `Xoshiro256++` PRNG, `Linear`, `Embedding`, `LayerNorm`, `RmsNorm`, `Dropout`, `Conv2d`, `MultiHeadAttention`, `TransformerBlock` |
| [`crates/as-optim`](crates/as-optim) | `Sgd`, `Adam`, `AdamW`, `StepLr`, `CosineAnnealingLr`, `clip_grad_norm`, `clip_grad_value` |
| [`crates/as-train`](crates/as-train) | `Dataset`, `DataLoader`, CSV dataset I/O & splitting, metrics, checksummed `ASCK` checkpoints |
| [`crates/as-io`](crates/as-io) | Bounded JSON parser, security-hardened **Safetensors** reader/writer, `ModelConfig` |
| [`crates/as-runtime`](crates/as-runtime) | `InferenceSession` with graph optimization, Safetensors binding, profiling, dynamic batching & streaming |
| [`crates/as-llm`](crates/as-llm) | Llama/Qwen2/Mistral/SmolLM decoder, GQA, RoPE, KV-Cache, HuggingFace BPE `tokenizer.json`, Int8 quantization |
| [`crates/as-vision`](crates/as-vision) | Validated `ImageBuffer`, bilinear resize, CHW normalization, `PatchEmbed`, `VisionTransformer`, `CnnClassifier` |
| [`crates/as-accel`](crates/as-accel) | Hardware discovery, MSL / WGSL / CUDA C++ kernel sources, cross-backend numerical consistency harness |
| [`crates/as-wasm`](crates/as-wasm) | Zero-dependency WebAssembly (`wasm32-unknown-unknown`) & native C-ABI (`cdylib`) bridge |
| [`crates/as-cli`](crates/as-cli) | CLI: `info`, `make-dataset`, `train`, `predict`, `demo-chat`, `run-llm`, `inspect-safetensors` |
| [`crates/as-bench`](crates/as-bench) | Reproducible zero-dependency statistical benchmark runner |
| [`sdk/typescript`](sdk/typescript) | Published npm package **`tarsix`** (Node.js & Browser WASM + TypeScript types) |
| [`sdk/python`](sdk/python) & [`include/tarsix.h`](include/tarsix.h) | Zero-dependency Python `ctypes` SDK & C/C++ ABI header |

---

## Quick Start

### Rust CLI & Benchmarks
```bash
# Run all 37 unit, numerical, gradient-check, and security tests
cargo test --workspace

# Run reproducible CPU benchmarks
cargo run --release -p as-bench

# Train a neural network on any CSV dataset and run live predictions
cargo run --release -p as-cli -- make-dataset /tmp/data.csv 240
cargo run --release -p as-cli -- train /tmp/data.csv /tmp/model.asck 80 24
cargo run --release -p as-cli -- predict /tmp/model.asck "0.1,0.1"

# Train a mini Causal Transformer from scratch & chat
cargo run --release -p as-cli -- demo-chat $'S: merhaba\nC:'

# Run any compatible HuggingFace Safetensors LLM (SmolLM2 / Qwen2.5 / Llama / Mistral)
cargo run --release -p as-cli -- run-llm /path/to/hf_model "Artificial intelligence is" 40 0.7
```

### JavaScript / TypeScript (`npm install tarsix`)
```js
import { createEngine, Tensor } from "tarsix";

const engine = await createEngine();
const a = Tensor.fromF32(engine, [1, 2, 3, 4], [2, 2]);
const b = Tensor.fromF32(engine, [5, 6, 7, 8], [2, 2]);
console.log(a.matmul(b).toF32Array()); // Float32Array [ 19, 22, 43, 50 ]
```

### Python (`ctypes`, Zero `pip` Dependencies)
```bash
cargo build -p as-wasm --release
python3 sdk/python/tarsix.py
```

---

## Verified CPU Benchmarks (`aarch64` / macOS / `--release`, 100% Safe Rust)

| Benchmark | Median (ms) | Throughput |
| :--- | ---: | ---: |
| `add_f32_contiguous_1M` | 0.235 | 50.98 GB/s |
| `add_f32_broadcast_1024x1024` | 0.149 | 56.38 GB/s |
| `matmul_f32_256x256` | 0.492 | 68.15 GFLOP/s |
| `matmul_f32_512x512` | 3.870 | 69.37 GFLOP/s |

---

## Security & Architecture Decisions

- [SECURITY.md](SECURITY.md)
- [ADR-001: Language Selection and Modular Workspace Architecture](docs/adr/ADR-001-language-and-workspace-architecture.md)
- [ADR-002: Memory Safety, Panic-Free APIs, and Untrusted-Input Security Model](docs/adr/ADR-002-memory-safety-and-security-model.md)
- [ADR-003: Tensor Layout, Zero-Copy Views, and Independent Half-Precision Types](docs/adr/ADR-003-tensor-layout-and-dtype-system.md)
- [ADR-004: Computational Graph IR, Operation Fusion, and Panic-Free Autodiff Tape](docs/adr/ADR-004-graph-and-autograd-architecture.md)

## License

Apache-2.0 — see [LICENSE](LICENSE).
