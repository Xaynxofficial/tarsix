//! Real Apple Silicon Metal GPU backend (`as-metal`).
//!
//! Isolates the narrow `unsafe` FFI boundary to Apple's `Metal.framework` inside this
//! dedicated driver crate (per ADR-002), while all core engine crates remain `#![forbid(unsafe_code)]`.

use std::sync::Arc;

use as_backend::{Backend, BinaryOp, ReduceOp, Storage, UnaryOp};
use as_core::{
    BF16, DType, Device, DeviceCapabilities, Error, F16, Layout, MemoryLimits, Result, Shape,
};
use as_cpu::{CpuBackend, CpuBuffer, CpuStorage};

#[cfg(target_os = "macos")]
mod ffi {
    use core::ffi::c_void;

    #[repr(C)]
    pub(crate) struct AsMetalContext {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        pub(crate) fn as_metal_create(name_buf: *mut u8, name_buf_len: usize) -> *mut AsMetalContext;
        pub(crate) fn as_metal_destroy(ctx: *mut AsMetalContext);
        pub(crate) fn as_metal_gemm_f32(
            ctx: *mut AsMetalContext,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: u32,
            k: u32,
            n: u32,
        ) -> i32;
        pub(crate) fn as_metal_add_f32(
            ctx: *mut AsMetalContext,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            count: u32,
        ) -> i32;
    }

    pub(crate) struct MetalHandle {
        pub(crate) raw: *mut AsMetalContext,
        pub(crate) device_name: String,
    }

    // SAFETY: MTLDevice, MTLCommandQueue, and MTLComputePipelineState are thread-safe in Metal.
    unsafe impl Send for MetalHandle {}
    unsafe impl Sync for MetalHandle {}

    impl MetalHandle {
        pub(crate) fn new() -> Option<Self> {
            let mut buf = [0u8; 256];
            // SAFETY: `buf` is valid writable memory of length 256.
            let raw = unsafe { as_metal_create(buf.as_mut_ptr(), buf.len()) };
            if raw.is_null() {
                return None;
            }
            let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            let device_name = String::from_utf8_lossy(&buf[..nul_pos]).into_owned();
            let _ = core::mem::size_of::<*mut c_void>();
            Some(Self { raw, device_name })
        }
    }

    impl Drop for MetalHandle {
        fn drop(&mut self) {
            // SAFETY: `self.raw` was allocated by `as_metal_create` and is destroyed once.
            unsafe { as_metal_destroy(self.raw) };
        }
    }
}

/// Real Apple Silicon Metal GPU execution backend implementing [`as_backend::Backend`].
#[derive(Clone)]
pub struct MetalBackend {
    #[cfg(target_os = "macos")]
    handle: Arc<ffi::MetalHandle>,
    #[cfg(not(target_os = "macos"))]
    _unused: Arc<()>,
    cpu_fallback: CpuBackend,
    limits: MemoryLimits,
}

impl MetalBackend {
    /// Initializes the system default Apple Silicon Metal GPU device.
    pub fn new() -> Result<Self> {
        Self::with_limits(MemoryLimits::default())
    }

    /// Initializes the Metal GPU backend with custom [`MemoryLimits`].
    pub fn with_limits(limits: MemoryLimits) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let handle = ffi::MetalHandle::new().ok_or_else(|| Error::InvalidArgument {
                context: "MetalBackend::new",
                reason: "no Metal GPU device available on this host".to_owned(),
            })?;
            Ok(Self {
                handle: Arc::new(handle),
                cpu_fallback: CpuBackend::with_limits(limits),
                limits,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = limits;
            Err(Error::InvalidArgument {
                context: "MetalBackend::new",
                reason: "Metal GPU backend is only available on macOS".to_owned(),
            })
        }
    }

    /// Returns the detected Metal GPU device name (e.g., `"Apple M3 Pro"`).
    pub fn device_name(&self) -> &str {
        #[cfg(target_os = "macos")]
        {
            &self.handle.device_name
        }
        #[cfg(not(target_os = "macos"))]
        {
            "unavailable"
        }
    }

    /// Executes 2D `f32` matrix multiplication `[m, k] @ [k, n]` directly on the Metal GPU.
    pub fn gemm_f32(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if a.len() != m.saturating_mul(k) || b.len() != k.saturating_mul(n) {
            return Err(Error::BufferLengthMismatch {
                expected: m.saturating_mul(k),
                actual: a.len(),
            });
        }
        let out_len = m.checked_mul(n).ok_or(Error::ArithmeticOverflow {
            context: "MetalBackend::gemm_f32",
        })?;
        self.limits.check_allocation(out_len, DType::F32)?;
        if out_len == 0 || k == 0 {
            return Ok(vec![0.0f32; out_len]);
        }

        #[cfg(target_os = "macos")]
        {
            let m32 = u32::try_from(m).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::gemm_f32 m",
            })?;
            let k32 = u32::try_from(k).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::gemm_f32 k",
            })?;
            let n32 = u32::try_from(n).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::gemm_f32 n",
            })?;
            let mut c = vec![0.0f32; out_len];
            // SAFETY: Slice lengths were verified above to equal m*k, k*n, and m*n.
            let rc = unsafe {
                ffi::as_metal_gemm_f32(
                    self.handle.raw,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    m32,
                    k32,
                    n32,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::gemm_f32",
                    reason: format!("Metal GPU kernel failed with status {rc}"),
                });
            }
            Ok(c)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::gemm_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Executes contiguous `f32` element-wise addition directly on the Metal GPU.
    pub fn add_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(Error::BufferLengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        self.limits.check_allocation(a.len(), DType::F32)?;
        if a.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(target_os = "macos")]
        {
            let count = u32::try_from(a.len()).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::add_f32",
            })?;
            let mut c = vec![0.0f32; a.len()];
            // SAFETY: Slice lengths are verified equal to `count`.
            let rc = unsafe {
                ffi::as_metal_add_f32(
                    self.handle.raw,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    count,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::add_f32",
                    reason: format!("Metal GPU add kernel failed with status {rc}"),
                });
            }
            Ok(c)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::add_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }
}

impl Backend for MetalBackend {
    type Storage = CpuStorage;

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            device: Device::Metal(0),
            name: format!("Apple Metal GPU ({})", self.device_name()),
            native_f16: true,
            native_bf16: true,
            deterministic: true,
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
        if op == BinaryOp::Add
            && lhs.dtype() == DType::F32
            && rhs.dtype() == DType::F32
            && lhs_layout.is_contiguous()
            && rhs_layout.is_contiguous()
            && lhs_layout.shape() == rhs_layout.shape()
        {
            let a = self.to_f32_vec(lhs, lhs_layout)?;
            let b = self.to_f32_vec(rhs, rhs_layout)?;
            let out = self.add_f32(&a, &b)?;
            return Ok(CpuStorage::from_buffer(CpuBuffer::F32(out)));
        }
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

    fn matmul(
        &self,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Shape)> {
        // Dispatch 2D F32 MatMul directly to Apple Silicon Metal GPU
        if lhs.dtype() == DType::F32
            && rhs.dtype() == DType::F32
            && lhs_layout.rank() == 2
            && rhs_layout.rank() == 2
        {
            let m = lhs_layout.shape().dim(0)?;
            let k = lhs_layout.shape().dim(1)?;
            let k2 = rhs_layout.shape().dim(0)?;
            let n = rhs_layout.shape().dim(1)?;
            if k != k2 {
                return Err(Error::ShapeMismatch {
                    context: "MetalBackend::matmul",
                    expected: lhs_layout.shape().dims().to_vec(),
                    actual: rhs_layout.shape().dims().to_vec(),
                });
            }
            let a = self.to_f32_vec(lhs, lhs_layout)?;
            let b = self.to_f32_vec(rhs, rhs_layout)?;
            let c = self.gemm_f32(&a, &b, m, k, n)?;
            return Ok((
                CpuStorage::from_buffer(CpuBuffer::F32(c)),
                Shape::new(vec![m, n])?,
            ));
        }
        self.cpu_fallback.matmul(lhs, lhs_layout, rhs, rhs_layout)
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
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use as_accel::verify_backend_consistency;

    use super::*;

    #[test]
    fn metal_gpu_executes_and_matches_cpu_reference() {
        let Ok(metal) = MetalBackend::new() else {
            // Gracefully skip if running inside a headless/virtualized sandbox without GPU access
            eprintln!("Metal GPU unavailable in current environment; skipping GPU dispatch test.");
            return;
        };
        let cpu = CpuBackend::new();
        let report = verify_backend_consistency(&cpu, &metal, 1e-4, 1e-4).unwrap();
        eprintln!(
            "Metal GPU verified: device=\"{}\", ops_checked={}, max_abs_error={:.2e}",
            metal.device_name(),
            report.ops_checked,
            report.max_abs_error
        );
        assert!(
            report.passed,
            "Metal GPU ({}) diverged from CPU reference: {report:?}",
            metal.device_name()
        );
    }
}
