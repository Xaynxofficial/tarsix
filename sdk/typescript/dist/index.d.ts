export type DType = "f32" | "f64" | "f16" | "bf16" | "i32" | "i64";
export type Device = "cpu" | `metal:${number}` | `cuda:${number}` | `webgpu:${number}`;

export interface MemoryLimits {
  readonly maxTensorBytes: number;
  readonly maxRank: number;
}

export interface TensorHandle {
  readonly id: number;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly device: Device;
}

export interface NativeEngineBridge {
  zeros(shape: readonly number[], dtype: DType, device: Device): TensorHandle;
  ones(shape: readonly number[], dtype: DType, device: Device): TensorHandle;
  fromF32(data: Float32Array, shape: readonly number[], device: Device): TensorHandle;
  toF32(handle: TensorHandle): Float32Array;
  add(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  sub(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  mul(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  div(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  matmul(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  transpose(src: TensorHandle, axisA: number, axisB: number): TensorHandle;
  reshape(src: TensorHandle, newShape: readonly number[]): TensorHandle;
  sum(src: TensorHandle): TensorHandle;
  mean(src: TensorHandle): TensorHandle;
  dispose(handle: TensorHandle): void;
}

export declare class WasmEngineBridge implements NativeEngineBridge {
  static instantiate(wasmBytes: BufferSource): Promise<WasmEngineBridge>;
  zeros(shape: readonly number[], dtype?: DType, device?: Device): TensorHandle;
  ones(shape: readonly number[], dtype?: DType, device?: Device): TensorHandle;
  fromF32(data: Float32Array, shape: readonly number[], device?: Device): TensorHandle;
  toF32(handle: TensorHandle): Float32Array;
  add(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  sub(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  mul(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  div(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  matmul(lhs: TensorHandle, rhs: TensorHandle): TensorHandle;
  transpose(src: TensorHandle, axisA: number, axisB: number): TensorHandle;
  reshape(src: TensorHandle, newShape: readonly number[]): TensorHandle;
  sum(src: TensorHandle): TensorHandle;
  mean(src: TensorHandle): TensorHandle;
  dispose(handle: TensorHandle): void;
}

export declare class Tensor {
  static zeros(bridge: NativeEngineBridge, shape: readonly number[], dtype?: DType, device?: Device): Tensor;
  static ones(bridge: NativeEngineBridge, shape: readonly number[], dtype?: DType, device?: Device): Tensor;
  static fromF32(
    bridge: NativeEngineBridge,
    data: Float32Array | readonly number[],
    shape: readonly number[],
    device?: Device,
  ): Tensor;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly device: Device;
  readonly handle: TensorHandle;
  add(other: Tensor): Tensor;
  sub(other: Tensor): Tensor;
  mul(other: Tensor): Tensor;
  div(other: Tensor): Tensor;
  matmul(other: Tensor): Tensor;
  transpose(axisA: number, axisB: number): Tensor;
  reshape(newShape: readonly number[]): Tensor;
  sum(): Tensor;
  mean(): Tensor;
  toF32Array(): Float32Array;
  dispose(): void;
}

export declare function createEngine(customWasmBytes?: BufferSource): Promise<WasmEngineBridge>;
