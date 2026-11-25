export class Tensor {
  #bridge;
  #handle;
  #disposed = false;

  constructor(bridge, handle) {
    this.#bridge = bridge;
    this.#handle = handle;
  }

  static zeros(bridge, shape, dtype = "f32", device = "cpu") {
    return new Tensor(bridge, bridge.zeros(shape, dtype, device));
  }

  static ones(bridge, shape, dtype = "f32", device = "cpu") {
    return new Tensor(bridge, bridge.ones(shape, dtype, device));
  }

  static fromF32(bridge, data, shape, device = "cpu") {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    return new Tensor(bridge, bridge.fromF32(arr, shape, device));
  }

  get shape() {
    return this.#handle.shape;
  }

  get dtype() {
    return this.#handle.dtype;
  }

  get device() {
    return this.#handle.device;
  }

  get handle() {
    this.#assertAlive();
    return this.#handle;
  }

  add(other) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.add(this.#handle, other.handle));
  }

  sub(other) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.sub(this.#handle, other.handle));
  }

  mul(other) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.mul(this.#handle, other.handle));
  }

  div(other) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.div(this.#handle, other.handle));
  }

  matmul(other) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.matmul(this.#handle, other.handle));
  }

  transpose(axisA, axisB) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.transpose(this.#handle, axisA, axisB));
  }

  reshape(newShape) {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.reshape(this.#handle, newShape));
  }

  sum() {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.sum(this.#handle));
  }

  mean() {
    this.#assertAlive();
    return new Tensor(this.#bridge, this.#bridge.mean(this.#handle));
  }

  toF32Array() {
    this.#assertAlive();
    return this.#bridge.toF32(this.#handle);
  }

  dispose() {
    if (!this.#disposed) {
      this.#bridge.dispose(this.#handle);
      this.#disposed = true;
    }
  }

  #assertAlive() {
    if (this.#disposed) {
      throw new Error("Tensor has already been disposed");
    }
  }
}
