import { Tensor } from "./tensor.js";
import { WasmEngineBridge } from "./wasm_bridge.js";

/**
 * Loads the bundled Rust WebAssembly AI engine in either Node.js or Web Browsers.
 */
export async function createEngine(customWasmBytes) {
  if (customWasmBytes) {
    return WasmEngineBridge.instantiate(customWasmBytes);
  }
  const wasmUrl = new URL("../wasm/as_wasm.wasm", import.meta.url);
  if (typeof process !== "undefined" && process.versions?.node) {
    const { readFile } = await import("node:fs/promises");
    const bytes = await readFile(wasmUrl);
    return WasmEngineBridge.instantiate(bytes);
  }
  const resp = await fetch(wasmUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch as_wasm.wasm: ${resp.status}`);
  }
  return WasmEngineBridge.instantiate(await resp.arrayBuffer());
}

export { Tensor, WasmEngineBridge };
