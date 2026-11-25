// Live end-to-end Node.js verification test executing Rust `as_wasm.wasm` directly.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.resolve(__dirname, "../../target/wasm32-unknown-unknown/release/as_wasm.wasm");
const wasmBytes = fs.readFileSync(wasmPath);

const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const w = instance.exports;

function createF32(arr, shape) {
  const dataPtr = w.as_alloc_f32(arr.length);
  new Float32Array(w.memory.buffer, dataPtr, arr.length).set(arr);
  const shapePtr = w.as_alloc_u32(shape.length);
  new Uint32Array(w.memory.buffer, shapePtr, shape.length).set(shape);
  const id = w.as_tensor_from_f32(dataPtr, arr.length, shapePtr, shape.length);
  w.as_free_f32(dataPtr, arr.length);
  w.as_free_u32(shapePtr, shape.length);
  assert.ok(id > 0, "tensor creation failed");
  return id;
}

function readF32(id) {
  const n = w.as_tensor_numel(id);
  const ptr = w.as_alloc_f32(n);
  const rc = w.as_tensor_read_f32(id, ptr, n);
  assert.equal(rc, 0);
  const out = new Float32Array(w.memory.buffer, ptr, n).slice();
  w.as_free_f32(ptr, n);
  return Array.from(out);
}

// 1. Matrix multiplication: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
const a = createF32([1, 2, 3, 4], [2, 2]);
const b = createF32([5, 6, 7, 8], [2, 2]);
const c = w.as_tensor_matmul(a, b);
assert.deepEqual(readF32(c), [19, 22, 43, 50]);

// 2. Broadcast bias addition: [[19, 22], [43, 50]] + [[1, -2]] = [[20, 20], [44, 48]]
const bias = createF32([1, -2], [1, 2]);
const biased = w.as_tensor_add(c, bias);
assert.deepEqual(readF32(biased), [20, 20, 44, 48]);

// 3. Mean reduction: (20 + 20 + 44 + 48) / 4 = 33
const m = w.as_tensor_mean(biased);
assert.deepEqual(readF32(m), [33]);

for (const id of [a, b, c, bias, biased, m]) {
  w.as_tensor_dispose(id);
}

console.log(
  `WASM live test PASSED (${(wasmBytes.length / 1024).toFixed(1)} KiB binary): matmul + broadcast add + mean verified in Node.js`,
);
