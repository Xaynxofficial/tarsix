import type { Device, DType, NativeEngineBridge, TensorHandle } from "./types.js";

/**
 * Strongly typed N-dimensional Tensor backed by the Rust core engine
 * over N-API (Node.js) or WebAssembly (Browser).
 */
export class Tensor {
  readonly #bridge: NativeEngineBridge;
  readonly #handle: TensorHandle;
  #disposed = false;

  private constructor(bridge: NativeEngineBridge, handle: TensorHandle) {
    this.#bridge = bridge;
    this.#handle = handle;
  }

  static zeros(
    bridge: NativeEngineBridge,
    shape: readonly number[],
    dtype: DType = "f32",
    device: Device = "cpu",
  ): Tensor {
    return new Tensor(bridge, bridge.zeros(shape, dtype, device));
  }

  static ones(
    bridge: NativeEngineBridge,
    shape: readonly number[],
    dtype: DType = "f32",
    device: Device = "cpu",
  ): Tensor {
    return new Tensor(bridge, bridge.ones(shape, dtype, device));
  }

  static fromF32(
    bridge: NativeEngineBridge,
    data: Float32Array | readonly number[],
    shape: readonly number[],
    device: Device = "cpu",
  ): Tensor {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    return new Tensor(bridge, bridge.fromF32(arr, shape, device));
  }

  get shape(): readonly number[] {
    return this.#handle.shape;
  }

  get dtype(): DType {
    return this.#handle.dtype;
  }

  get device(): Device {
    return this.#handle.device;
  }

  get handle(): TensorHandle {
    this.#assertAlive();
    return this.#handle;
  }

  add(other: Tensor): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.add(this.#handle, other.handle));
  }

  sub(other: Tensor): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.sub(this.#handle, other.handle));
  }

  mul(other: Tensor): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.mul(this.#handle, other.handle));
  }

  div(other: Tensor): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.div(this.#handle, other.handle));
  }

  matmul(other: Tensor): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.matmul(this.#handle, other.handle));
  }

  transpose(axisA: number, axisB: number): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.transpose(this.#handle, axisA, axisB));
  }

  reshape(newShape: readonly number[]): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.reshape(this.#handle, newShape));
  }

  sum(): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.sum(this.#handle));
  }

  mean(): Tensor {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.mean(this.#handle));
  }

  toF32Array(): Float32Array {
    this.#assertAlive();
    return this.#bridge.toF32(this.#handle);
  }

  dispose(): void {
    if (!this.#disposed) {
      this.#bridge.dispose(this.#handle);
      this.#disposed = true;
    }
  }

  #assertAlive(): void {
    if (this.#disposed) {
      throw new Error("Tensor has already been disposed");
    }
  }
}
