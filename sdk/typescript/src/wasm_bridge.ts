import type { Device, DType, NativeEngineBridge, TensorHandle } from "./types.js";

interface AsWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  as_alloc_f32(len: number): number;
  as_free_f32(ptr: number, len: number): void;
  as_alloc_u32(len: number): number;
  as_free_u32(ptr: number, len: number): void;
  as_tensor_zeros(shapePtr: number, rank: number): number;
  as_tensor_ones(shapePtr: number, rank: number): number;
  as_tensor_from_f32(dataPtr: number, dataLen: number, shapePtr: number, rank: number): number;
  as_tensor_add(a: number, b: number): number;
  as_tensor_sub(a: number, b: number): number;
  as_tensor_mul(a: number, b: number): number;
  as_tensor_div(a: number, b: number): number;
  as_tensor_matmul(a: number, b: number): number;
  as_tensor_transpose(a: number, axisA: number, axisB: number): number;
  as_tensor_reshape(a: number, shapePtr: number, rank: number): number;
  as_tensor_sum(a: number): number;
  as_tensor_mean(a: number): number;
  as_tensor_numel(id: number): number;
  as_tensor_read_f32(id: number, outPtr: number, outLen: number): number;
  as_tensor_dispose(id: number): void;
}

/**
 * Live WebAssembly bridge connecting `@as-engine/sdk` to the compiled Rust `as_wasm.wasm` binary
 * in both Node.js and Web Browsers.
 */
export class WasmEngineBridge implements NativeEngineBridge {
  readonly #wasm: AsWasmExports;

  private constructor(wasm: AsWasmExports) {
    this.#wasm = wasm;
  }

  static async instantiate(wasmBytes: BufferSource): Promise<WasmEngineBridge> {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return new WasmEngineBridge(instance.exports as AsWasmExports);
  }

  #withShape<T>(shape: readonly number[], fn: (ptr: number, rank: number) => T): T {
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

  zeros(shape: readonly number[], dtype: DType, device: Device): TensorHandle {
    const id = this.#withShape(shape, (p, r) => this.#wasm.as_tensor_zeros(p, r));
    if (id === 0) throw new Error("WASM as_tensor_zeros failed");
    return { id, shape: [...shape], dtype, device };
  }

  ones(shape: readonly number[], dtype: DType, device: Device): TensorHandle {
    const id = this.#withShape(shape, (p, r) => this.#wasm.as_tensor_ones(p, r));
    if (id === 0) throw new Error("WASM as_tensor_ones failed");
    return { id, shape: [...shape], dtype, device };
  }

  fromF32(data: Float32Array, shape: readonly number[], device: Device): TensorHandle {
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

  toF32(handle: TensorHandle): Float32Array {
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

  add(lhs: TensorHandle, rhs: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_add(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_add failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  sub(lhs: TensorHandle, rhs: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_sub(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_sub failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  mul(lhs: TensorHandle, rhs: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_mul(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_mul failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  div(lhs: TensorHandle, rhs: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_div(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_div failed");
    return { id, shape: [...lhs.shape], dtype: lhs.dtype, device: lhs.device };
  }

  matmul(lhs: TensorHandle, rhs: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_matmul(lhs.id, rhs.id);
    if (id === 0) throw new Error("WASM as_tensor_matmul failed");
    const m = lhs.shape[lhs.shape.length - 2] ?? 0;
    const n = rhs.shape[rhs.shape.length - 1] ?? 0;
    return { id, shape: [m, n], dtype: lhs.dtype, device: lhs.device };
  }

  transpose(src: TensorHandle, axisA: number, axisB: number): TensorHandle {
    const id = this.#wasm.as_tensor_transpose(src.id, axisA, axisB);
    if (id === 0) throw new Error("WASM as_tensor_transpose failed");
    const shape = [...src.shape];
    const tmp = shape[axisA]!;
    shape[axisA] = shape[axisB]!;
    shape[axisB] = tmp;
    return { id, shape, dtype: src.dtype, device: src.device };
  }

  reshape(src: TensorHandle, newShape: readonly number[]): TensorHandle {
    const id = this.#withShape(newShape, (p, r) => this.#wasm.as_tensor_reshape(src.id, p, r));
    if (id === 0) throw new Error("WASM as_tensor_reshape failed");
    return { id, shape: [...newShape], dtype: src.dtype, device: src.device };
  }

  sum(src: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_sum(src.id);
    if (id === 0) throw new Error("WASM as_tensor_sum failed");
    return { id, shape: [], dtype: src.dtype, device: src.device };
  }

  mean(src: TensorHandle): TensorHandle {
    const id = this.#wasm.as_tensor_mean(src.id);
    if (id === 0) throw new Error("WASM as_tensor_mean failed");
    return { id, shape: [], dtype: src.dtype, device: src.device };
  }

  dispose(handle: TensorHandle): void {
    this.#wasm.as_tensor_dispose(handle.id);
  }
}
