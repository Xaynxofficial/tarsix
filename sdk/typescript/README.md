# `tarsix`

Zero-dependency, memory-safe **Rust + WebAssembly** AI Tensor Engine for **Node.js** and **Browsers**.

- **114 KiB** self-contained WebAssembly binary bundled — no Rust toolchain or Python required.
- Built on `#![forbid(unsafe_code)]` Rust tensor core with overflow-checked shapes, broadcasting, tiled MatMul, zero-copy transpose/reshape, and reductions.
- Full TypeScript definitions included out of the box.

## Install

```bash
npm install tarsix
```

## Quick Start

```js
import { createEngine, Tensor } from "tarsix";

const engine = await createEngine();

const a = Tensor.fromF32(engine, [1, 2, 3, 4], [2, 2]);
const b = Tensor.fromF32(engine, [5, 6, 7, 8], [2, 2]);
const bias = Tensor.fromF32(engine, [1, -2], [1, 2]);

const out = a.matmul(b).add(bias);
console.log(out.toF32Array()); // Float32Array(4) [ 20, 20, 44, 48 ]

a.dispose();
b.dispose();
bias.dispose();
out.dispose();
```

## License

Apache-2.0
