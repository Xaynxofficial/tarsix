import assert from "node:assert/strict";
import { createEngine, Tensor } from "./dist/index.js";

const engine = await createEngine();

const a = Tensor.fromF32(engine, [1, 2, 3, 4], [2, 2]);
const b = Tensor.fromF32(engine, [5, 6, 7, 8], [2, 2]);
const bias = Tensor.fromF32(engine, [1, -2], [1, 2]);

const c = a.matmul(b);
const out = c.add(bias);
const mean = out.mean();

assert.deepEqual(Array.from(c.toF32Array()), [19, 22, 43, 50]);
assert.deepEqual(Array.from(out.toF32Array()), [20, 20, 44, 48]);
assert.deepEqual(Array.from(mean.toF32Array()), [33]);

for (const t of [a, b, bias, c, out, mean]) {
  t.dispose();
}

console.log("npm package test PASSED: createEngine() + Tensor API verified!");
