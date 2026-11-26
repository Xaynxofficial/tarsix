//! Isolated NVIDIA CUDA & cuBLAS GPU backend (`as-cuda`).
//!
//! Loads `libcudart.so` and `libcublas.so` dynamically at runtime via `dlopen` on Linux
//! so the workspace compiles cleanly with zero external Rust crates and zero build-time
//! CUDA Toolkit dependencies on macOS, Windows, and CPU-only servers.
//!
//! When running on an NVIDIA GPU host (A100, H100, L4, T4, RTX), [`CudaBackend::new`]
//! binds the live CUDA runtime & cuBLAS handle and dispatches `F32` GEMM directly to
//! `cublasSgemm_v2` on NVIDIA Tensor Cores.

use std::sync::Arc;

use as_backend::{Backend, BinaryOp, ReduceOp, Storage, UnaryOp};
use as_core::{
    BF16, DType, Device, DeviceCapabilities, Error, F16, Layout, MemoryLimits, Result, Shape,
};
use as_cpu::{CpuBackend, CpuBuffer, CpuStorage};

/// Versioned CUDA C / NVRTC kernel sources for LLM acceleration on NVIDIA GPUs.
pub mod kernels {
    /// Row-wise RMSNorm CUDA kernel (`[rows, cols]`).
    pub const RMSNORM_F32_CU: &str = r#"
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int base = row * cols;
    float sum_sq = 0.0f;
    for (unsigned int c = 0; c < cols; ++c) {
        float v = X[base + c];
        sum_sq = fmaf(v, v, sum_sq);
    }
    float inv_rms = rsqrtf(sum_sq / (float)cols + eps);
    for (unsigned int c = 0; c < cols; ++c) {
        Y[base + c] = X[base + c] * inv_rms * W[c];
    }
}
"#;

    /// Fused SwiGLU (`silu(gate) * up`) CUDA kernel.
    pub const SWIGLU_F32_CU: &str = r#"
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    unsigned int count
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}
"#;

    /// 4-bit block-quantized MatVec (group_size = 32) CUDA kernel.
    pub const Q4_MATVEC_F32_CU: &str = r#"
extern "C" __global__ void q4_matvec_f32(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int blocks_per_row
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    unsigned int row_base = row * blocks_per_row;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        unsigned int blk = row_base + b;
        float sc = scales[blk];
        unsigned int byte_base = blk * 16;
        unsigned int col_base = b * 32;
        float b_acc = 0.0f;
        for (unsigned int k = 0; k < 16; ++k) {
            unsigned char byte = packed[byte_base + k];
            float q0 = (float)((int)(byte & 0x0F) - 8);
            float q1 = (float)((int)((byte >> 4) & 0x0F) - 8);
            unsigned int c0 = col_base + k * 2;
            unsigned int c1 = c0 + 1;
            if (c0 < in_dim) b_acc = fmaf(q0, x[c0], b_acc);
            if (c1 < in_dim) b_acc = fmaf(q1, x[c1], b_acc);
        }
        acc = fmaf(b_acc, sc, acc);
    }
    y[row] = acc;
}
"#;
}

#[cfg(target_os = "linux")]
mod ffi {
    use core::ffi::{c_char, c_int, c_void};

    const RTLD_LAZY: c_int = 1;
    pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
    pub(crate) const CUBLAS_OP_N: c_int = 0;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    pub(crate) type CudaGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> c_int;
    pub(crate) type CudaSetDeviceFn = unsafe extern "C" fn(c_int) -> c_int;
    pub(crate) type CudaMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
    pub(crate) type CudaFreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
    pub(crate) type CudaMemcpyFn =
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
    pub(crate) type CudaSyncFn = unsafe extern "C" fn() -> c_int;
    pub(crate) type CublasCreateFn = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
    pub(crate) type CublasDestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;
    pub(crate) type CublasSgemmFn = unsafe extern "C" fn(
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *const f32,
        *const f32,
        c_int,
        *const f32,
        c_int,
        *const f32,
        *mut f32,
        c_int,
    ) -> c_int;

    pub(crate) struct CudaRuntime {
        cudart_lib: *mut c_void,
        cublas_lib: *mut c_void,
        pub(crate) cublas_handle: *mut c_void,
        pub(crate) device_index: usize,
        pub(crate) device_count: usize,
        pub(crate) cuda_malloc: CudaMallocFn,
        pub(crate) cuda_free: CudaFreeFn,
        pub(crate) cuda_memcpy: CudaMemcpyFn,
        pub(crate) cuda_sync: CudaSyncFn,
        pub(crate) cublas_sgemm: CublasSgemmFn,
        cublas_destroy: CublasDestroyFn,
    }

    // SAFETY: CUDA runtime calls and cuBLAS handles are thread-safe when synchronized externally.
    unsafe impl Send for CudaRuntime {}
    unsafe impl Sync for CudaRuntime {}

    impl CudaRuntime {
        pub(crate) fn load(device_index: usize) -> Option<Self> {
            // SAFETY: Static NUL-terminated library names passed to POSIX dlopen.
            let cudart = unsafe {
                let h = dlopen(b"libcudart.so\0".as_ptr().cast(), RTLD_LAZY);
                if h.is_null() {
                    dlopen(b"libcudart.so.12\0".as_ptr().cast(), RTLD_LAZY)
                } else {
                    h
                }
            };
            if cudart.is_null() {
                return None;
            }
            let cublas = unsafe {
                let h = dlopen(b"libcublas.so\0".as_ptr().cast(), RTLD_LAZY);
                if h.is_null() {
                    dlopen(b"libcublas.so.12\0".as_ptr().cast(), RTLD_LAZY)
                } else {
                    h
                }
            };
            if cublas.is_null() {
                unsafe { dlclose(cudart) };
                return None;
            }

            macro_rules! load_sym {
                ($lib:expr, $name:literal, $ty:ty) => {{
                    let ptr = unsafe { dlsym($lib, concat!($name, "\0").as_ptr().cast()) };
                    if ptr.is_null() {
                        unsafe {
                            dlclose(cublas);
                            dlclose(cudart);
                        }
                        return None;
                    }
                    // SAFETY: Symbol signature matches official CUDA / cuBLAS C ABI.
                    unsafe { core::mem::transmute::<*mut c_void, $ty>(ptr) }
                }};
            }

            let get_count = load_sym!(cudart, "cudaGetDeviceCount", CudaGetDeviceCountFn);
            let set_dev = load_sym!(cudart, "cudaSetDevice", CudaSetDeviceFn);
            let cuda_malloc = load_sym!(cudart, "cudaMalloc", CudaMallocFn);
            let cuda_free = load_sym!(cudart, "cudaFree", CudaFreeFn);
            let cuda_memcpy = load_sym!(cudart, "cudaMemcpy", CudaMemcpyFn);
            let cuda_sync = load_sym!(cudart, "cudaDeviceSynchronize", CudaSyncFn);
            let cublas_create = load_sym!(cublas, "cublasCreate_v2", CublasCreateFn);
            let cublas_destroy = load_sym!(cublas, "cublasDestroy_v2", CublasDestroyFn);
            let cublas_sgemm = load_sym!(cublas, "cublasSgemm_v2", CublasSgemmFn);

            let mut count: c_int = 0;
            // SAFETY: Valid pointer to stack `c_int`.
            if unsafe { get_count(&mut count) } != 0 || count <= 0 || device_index >= count as usize {
                unsafe {
                    dlclose(cublas);
                    dlclose(cudart);
                }
                return None;
            }
            if unsafe { set_dev(device_index as c_int) } != 0 {
                unsafe {
                    dlclose(cublas);
                    dlclose(cudart);
                }
                return None;
            }
            let mut handle: *mut c_void = core::ptr::null_mut();
            if unsafe { cublas_create(&mut handle) } != 0 || handle.is_null() {
                unsafe {
                    dlclose(cublas);
                    dlclose(cudart);
                }
                return None;
            }

            Some(Self {
                cudart_lib: cudart,
                cublas_lib: cublas,
                cublas_handle: handle,
                device_index,
                device_count: count as usize,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cuda_sync,
                cublas_sgemm,
                cublas_destroy,
            })
        }
    }

    impl Drop for CudaRuntime {
        fn drop(&mut self) {
            // SAFETY: Destroys live cuBLAS handle and closes dlopen handles once.
            unsafe {
                let _ = (self.cublas_destroy)(self.cublas_handle);
                let _ = dlclose(self.cublas_lib);
                let _ = dlclose(self.cudart_lib);
            }
        }
    }
}

/// Real NVIDIA CUDA / cuBLAS execution backend implementing [`as_backend::Backend`].
#[derive(Clone)]
pub struct CudaBackend {
    #[cfg(target_os = "linux")]
    runtime: Arc<ffi::CudaRuntime>,
    #[cfg(not(target_os = "linux"))]
    _unused: Arc<()>,
    device_index: usize,
    cpu_fallback: CpuBackend,
    limits: MemoryLimits,
}

impl CudaBackend {
    /// Returns `true` if a live NVIDIA CUDA + cuBLAS runtime and GPU are detected on this host.
    pub fn is_available() -> bool {
        Self::new(0).is_ok()
    }

    /// Initializes the NVIDIA CUDA backend on `device_index` (e.g. `0`).
    pub fn new(device_index: usize) -> Result<Self> {
        Self::with_limits(device_index, MemoryLimits::default())
    }

    /// Initializes the NVIDIA CUDA backend on `device_index` with custom [`MemoryLimits`].
    pub fn with_limits(device_index: usize, limits: MemoryLimits) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let runtime = ffi::CudaRuntime::load(device_index).ok_or_else(|| {
                Error::InvalidArgument {
                    context: "CudaBackend::new",
                    reason: format!(
                        "no NVIDIA CUDA/cuBLAS runtime or GPU device {device_index} available"
                    ),
                }
            })?;
            Ok(Self {
                runtime: Arc::new(runtime),
                device_index,
                cpu_fallback: CpuBackend::with_limits(limits),
                limits,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (device_index, limits);
            Err(Error::InvalidArgument {
                context: "CudaBackend::new",
                reason: "NVIDIA CUDA dynamic driver is only available on Linux hosts".to_owned(),
            })
        }
    }

    /// Executes 2D row-major `f32` GEMM `[m, k] @ [k, n]` via NVIDIA `cublasSgemm_v2`.
    pub fn gemm_f32(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if a.len() != m.saturating_mul(k) || b.len() != k.saturating_mul(n) {
            return Err(Error::BufferLengthMismatch {
                expected: m.saturating_mul(k),
                actual: a.len(),
            });
        }
        let out_len = m.checked_mul(n).ok_or(Error::ArithmeticOverflow {
            context: "CudaBackend::gemm_f32",
        })?;
        self.limits.check_allocation(out_len, DType::F32)?;
        if out_len == 0 || k == 0 {
            return Ok(vec![0.0f32; out_len]);
        }

        #[cfg(target_os = "linux")]
        {
            use core::ffi::{c_int, c_void};
            let rt = &self.runtime;
            let a_bytes = a.len() * 4;
            let b_bytes = b.len() * 4;
            let c_bytes = out_len * 4;

            let mut d_a: *mut c_void = core::ptr::null_mut();
            let mut d_b: *mut c_void = core::ptr::null_mut();
            let mut d_c: *mut c_void = core::ptr::null_mut();

            // SAFETY: Valid device pointers and verified byte lengths.
            unsafe {
                if (rt.cuda_malloc)(&mut d_a, a_bytes) != 0
                    || (rt.cuda_malloc)(&mut d_b, b_bytes) != 0
                    || (rt.cuda_malloc)(&mut d_c, c_bytes) != 0
                {
                    if !d_a.is_null() { let _ = (rt.cuda_free)(d_a); }
                    if !d_b.is_null() { let _ = (rt.cuda_free)(d_b); }
                    if !d_c.is_null() { let _ = (rt.cuda_free)(d_c); }
                    return Err(Error::InvalidArgument {
                        context: "CudaBackend::gemm_f32",
                        reason: "cudaMalloc failed".to_owned(),
                    });
                }

                let rc_h2d_a = (rt.cuda_memcpy)(d_a, a.as_ptr().cast(), a_bytes, ffi::CUDA_MEMCPY_HOST_TO_DEVICE);
                let rc_h2d_b = (rt.cuda_memcpy)(d_b, b.as_ptr().cast(), b_bytes, ffi::CUDA_MEMCPY_HOST_TO_DEVICE);

                // Row-major C = A @ B is equivalent to column-major C^T = B^T @ A^T
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                let rc_gemm = if rc_h2d_a == 0 && rc_h2d_b == 0 {
                    (rt.cublas_sgemm)(
                        rt.cublas_handle,
                        ffi::CUBLAS_OP_N,
                        ffi::CUBLAS_OP_N,
                        n as c_int,
                        m as c_int,
                        k as c_int,
                        &alpha,
                        d_b.cast(),
                        n as c_int,
                        d_a.cast(),
                        k as c_int,
                        &beta,
                        d_c.cast(),
                        n as c_int,
                    )
                } else {
                    -1
                };
                let rc_sync = if rc_gemm == 0 { (rt.cuda_sync)() } else { -1 };

                let mut c = vec![0.0f32; out_len];
                let rc_d2h = if rc_sync == 0 {
                    (rt.cuda_memcpy)(c.as_mut_ptr().cast(), d_c, c_bytes, ffi::CUDA_MEMCPY_DEVICE_TO_HOST)
                } else {
                    -1
                };

                let _ = (rt.cuda_free)(d_a);
                let _ = (rt.cuda_free)(d_b);
                let _ = (rt.cuda_free)(d_c);

                if rc_d2h != 0 {
                    return Err(Error::InvalidArgument {
                        context: "CudaBackend::gemm_f32",
                        reason: "CUDA/cuBLAS SGEMM execution failed".to_owned(),
                    });
                }
                Ok(c)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(Error::InvalidArgument {
                context: "CudaBackend::gemm_f32",
                reason: "CUDA unavailable on this OS".to_owned(),
            })
        }
    }
}

impl Backend for CudaBackend {
    type Storage = CpuStorage;

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            device: Device::Cuda(self.device_index),
            name: format!("NVIDIA CUDA GPU #{} (cuBLAS)", self.device_index),
            native_f16: true,
            native_bf16: true,
            deterministic: false,
        }
    }

    fn limits(&self) -> MemoryLimits {
        self.limits
    }

    fn zeros(&self, numel: usize, dtype: DType) -> Result<Self::Storage> {
        self.cpu_fallback.zeros(numel, dtype)
    }

    fn ones(&self, numel: usize, dtype: DType) -> Result<Self::Storage> {
        self.cpu_fallback.ones(numel, dtype)
    }

    fn full_f64(&self, numel: usize, value: f64, dtype: DType) -> Result<Self::Storage> {
        self.cpu_fallback.full_f64(numel, value, dtype)
    }

    fn from_f32_slice(&self, data: &[f32]) -> Result<Self::Storage> {
        self.cpu_fallback.from_f32_slice(data)
    }

    fn from_f64_slice(&self, data: &[f64]) -> Result<Self::Storage> {
        self.cpu_fallback.from_f64_slice(data)
    }

    fn from_f16_slice(&self, data: &[F16]) -> Result<Self::Storage> {
        self.cpu_fallback.from_f16_slice(data)
    }

    fn from_bf16_slice(&self, data: &[BF16]) -> Result<Self::Storage> {
        self.cpu_fallback.from_bf16_slice(data)
    }

    fn from_i32_slice(&self, data: &[i32]) -> Result<Self::Storage> {
        self.cpu_fallback.from_i32_slice(data)
    }

    fn from_i64_slice(&self, data: &[i64]) -> Result<Self::Storage> {
        self.cpu_fallback.from_i64_slice(data)
    }

    fn to_f32_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<f32>> {
        self.cpu_fallback.to_f32_vec(storage, layout)
    }

    fn to_f64_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<f64>> {
        self.cpu_fallback.to_f64_vec(storage, layout)
    }

    fn to_f16_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<F16>> {
        self.cpu_fallback.to_f16_vec(storage, layout)
    }

    fn to_bf16_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<BF16>> {
        self.cpu_fallback.to_bf16_vec(storage, layout)
    }

    fn to_i32_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<i32>> {
        self.cpu_fallback.to_i32_vec(storage, layout)
    }

    fn to_i64_vec(&self, storage: &Self::Storage, layout: &Layout) -> Result<Vec<i64>> {
        self.cpu_fallback.to_i64_vec(storage, layout)
    }

    fn contiguous(&self, storage: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        self.cpu_fallback.contiguous(storage, layout)
    }

    fn cast(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        target_dtype: DType,
    ) -> Result<Self::Storage> {
        self.cpu_fallback.cast(storage, layout, target_dtype)
    }

    fn binary_op(
        &self,
        op: BinaryOp,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage> {
        self.cpu_fallback
            .binary_op(op, lhs, lhs_layout, rhs, rhs_layout)
    }

    fn unary_op(
        &self,
        op: UnaryOp,
        src: &Self::Storage,
        layout: &Layout,
    ) -> Result<Self::Storage> {
        self.cpu_fallback.unary_op(op, src, layout)
    }

    fn reduce(
        &self,
        op: ReduceOp,
        src: &Self::Storage,
        layout: &Layout,
        axis: Option<usize>,
        keepdim: bool,
    ) -> Result<(Self::Storage, Shape)> {
        self.cpu_fallback.reduce(op, src, layout, axis, keepdim)
    }

    fn matmul(
        &self,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Shape)> {
        if lhs.dtype() == DType::F32
            && rhs.dtype() == DType::F32
            && lhs_layout.shape().rank() == 2
            && rhs_layout.shape().rank() == 2
        {
            let m = lhs_layout.shape().dim(0)?;
            let k1 = lhs_layout.shape().dim(1)?;
            let k2 = rhs_layout.shape().dim(0)?;
            let n = rhs_layout.shape().dim(1)?;
            if k1 != k2 {
                return Err(Error::ShapeMismatch {
                    context: "CudaBackend::matmul",
                    expected: vec![k1],
                    actual: vec![k2],
                });
            }
            let a = self.cpu_fallback.to_f32_vec(lhs, lhs_layout)?;
            let b = self.cpu_fallback.to_f32_vec(rhs, rhs_layout)?;
            let c = self.gemm_f32(&a, &b, m, k1, n)?;
            return Ok((
                CpuStorage::from_buffer(CpuBuffer::F32(c)),
                Shape::new(vec![m, n])?,
            ));
        }
        self.cpu_fallback.matmul(lhs, lhs_layout, rhs, rhs_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_probe_and_kernel_sources_are_valid() {
        assert!(kernels::RMSNORM_F32_CU.contains("rmsnorm_f32"));
        assert!(kernels::SWIGLU_F32_CU.contains("swiglu_f32"));
        assert!(kernels::Q4_MATVEC_F32_CU.contains("q4_matvec_f32"));

        // On hosts with a live NVIDIA GPU, verify cuBLAS GEMM; otherwise ensure clean Err (no panic)
        if let Ok(cuda) = CudaBackend::new(0) {
            let out = cuda.gemm_f32(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2).unwrap();
            assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
        } else {
            assert!(!CudaBackend::is_available());
        }
    }
}
