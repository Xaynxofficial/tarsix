export class WasmEngineBridge {
  #wasm;

  constructor(wasm) {
    this.#wasm = wasm;
  }

  static async instantiate(wasmBytes) {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return new WasmEngineBridge(instance.exports);
  }

  #withShape(shape, fn) {
    const rank = shape.length;
    const ptr = rank > 0 ? this.#wasm.as_alloc_u32(rank) : 0;
    try {
      if (rank > 0) {
        new Uint32Array(this.#wasm.memory.buffer, ptr, rank).set(shape);
      }
      return fn(ptr, rank);
    } finally {
      if (rank > 0) {
        this.#wasm.as_free_u32(ptr, rank);
      }
    }
  }

  zeros(shape, dtype = "f32", device = "cpu") {
    const id = this.#withShape(shape, (p, r) => this.#wasm.as_tensor_zeros(p, r));
    if (id === 0) throw new Error("WASM as_tensor_zeros failed");
    return { id, shape: [...shape], dtype, device };
  }

  ones(shape, dtype = "f32", device = "cpu") {
    const id = this.#withShape(shape, (p, r) => this.#wasm.as_tensor_ones(p, r));
    if (id === 0) throw new Error("WASM as_tensor_ones failed");
    return { id, shape: [...shape], dtype, device };
  }

  fromF32(data, shape, device = "cpu") {
    const dataPtr = data.length > 0 ? this.#wasm.as_alloc_f32(data.length) : 0;
    try {
      if (data.length > 0) {
        new Float32Array(this.#wasm.memory.buffer, dataPtr, data.length).set(data);
      }
      const id = this.#withShape(shape, (sp, r) =>
        this.#wasm.as_tensor_from_f32(dataPtr, data.length, sp, r),
      );
      if (id === 0) throw new Error("WASM as_tensor_from_f32 failed");
      return { id, shape: [...shape], dtype: "f32", device };
    } finally {
      if (data.length > 0) {
        this.#wasm.as_free_f32(dataPtr, data.length);
      }
    }
  }

  toF32(handle) {
    const n = this.#wasm.as_tensor_numel(handle.id);
    if (n === 0) return new Float32Array(0);
    const outPtr = this.#wasm.as_alloc_f32(n);
    try {
      const rc = this.#wasm.as_tensor_read_f32(handle.id, outPtr, n);
      if (rc !== 0) throw new Error(`WASM as_tensor_read_f32 failed with code ${rc}`);
      return new Float32Array(this.#wasm.memory.buffer, outPtr, n).slice();
    } finally {
      this.#wasm.as_free_f32(outPtr, n);
    }
  }

  add(lhs, rhs) {
    const id = this.#wasm.as_tensor_add(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_add failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  sub(lhs, rhs) {
    const id = this.#wasm.as_tensor_sub(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_sub failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  mul(lhs, rhs) {
    const id = this.#wasm.as_tensor_mul(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_mul failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  div(lhs, rhs) {
    const id = this.#wasm.as_tensor_div(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_div failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  matmul(lhs, rhs) {
    const id = this.#wasm.as_tensor_matmul(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_matmul failed");
    const m = lhs.shape[lhs.shape.length - 2] ?? 0;
    const n = rhs.shape[rhs.shape.length - 1] ?? 0;
    return { id, shape: [m, n], dtype: lhs.dtype, device: lhs.device };
  }

  transpose(src, axisA, axisB) {
    const id = this.#wasm.as_tensor_transpose(src.id, axisA, axisB);
    if (id === 0) throw new Error("WASM as_tensor_transpose failed");
    const shape = [...src.shape];
    const tmp = shape[axisA];
    shape[axisA] = shape[axisB];
    shape[axisB] = tmp;
    return { id, shape, dtype: src.dtype, device: src.device };
  }

  reshape(src, newShape) {
    const id = this.#withShape(newShape, (p, r) => this.#wasm.as_tensor_reshape(src.id, p, r));
    if (id === 0) throw new Error("WASM as_tensor_reshape failed");
    return { id, shape: [...newShape], dtype: src.dtype, device: src.device };
  }

  sum(src) {
    const id = this.#wasm.as_tensor_sum(src.id);
    if (id === 0) throw new Error("WASM as_tensor_sum failed");
    return { id, shape: [], dtype: src.dtype, device: src.device };
  }

  mean(src) {
    const id = this.#wasm.as_tensor_mean(src.id);
    if (id === 0) throw new Error("WASM as_tensor_mean failed");
    return { id, shape: [], dtype: src.dtype, device: src.device };
  }

  dispose(handle) {
    this.#wasm.as_tensor_dispose(handle.id);
  }
}
