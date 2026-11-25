/** Supported numeric tensor element data types matching Rust `as_core::DType`. */
export type DType = "f32" | "f64" | "f16" | "bf16" | "i32" | "i64";

/** Target execution device matching Rust `as_core::Device`. */
export type Device = "cpu" | `metal:${number}` | `cuda:${number}` | `webgpu:${number}`;

/** Security and resource limits enforced by the Rust core engine. */
export interface MemoryLimits {
  readonly maxTensorBytes: number;
  readonly maxRank: number;
}

/** Runtime session configuration matching Rust `as_runtime::RuntimeConfig`. */
export interface RuntimeConfig {
  readonly device?: Device;
  readonly limits?: MemoryLimits;
  readonly enableOptimization?: boolean;
  readonly enableProfiling?: boolean;
  readonly maxBatchSize?: number;
}

/** Autoregressive LLM sampling parameters matching Rust `as_llm::SamplingConfig`. */
export interface SamplingConfig {
  readonly temperature?: number;
  readonly topK?: number;
  readonly topP?: number;
  readonly repetitionPenalty?: number;
}

/** Execution telemetry returned by profiled inference runs. */
export interface InferenceProfile {
  readonly totalDurationUs: number;
  readonly executedNodes: number;
}

/** Opaque handle to a Rust-managed tensor buffer across N-API or WASM boundaries. */
export interface TensorHandle {
  readonly id: number;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly device: Device;
}

/**
 * Low-level bridge contract implemented by either:
 * - Node.js native N-API addon (`as-napi`)
 * - Browser WebAssembly / WebGPU module (`as-wasm`)
 */
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
