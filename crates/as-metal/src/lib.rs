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

    #[repr(C)]
    pub(crate) struct AsMetalBuffer {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub(crate) struct AsMetalBatch {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        pub(crate) fn as_metal_create(name_buf: *mut u8, name_buf_len: usize) -> *mut AsMetalContext;
        pub(crate) fn as_metal_destroy(ctx: *mut AsMetalContext);
        pub(crate) fn as_metal_buf_alloc(
            ctx: *mut AsMetalContext,
            byte_len: usize,
        ) -> *mut AsMetalBuffer;
        pub(crate) fn as_metal_buf_contents(buf: *mut AsMetalBuffer) -> *mut c_void;
        pub(crate) fn as_metal_buf_free(buf: *mut AsMetalBuffer);
        pub(crate) fn as_metal_batch_begin(ctx: *mut AsMetalContext) -> *mut AsMetalBatch;
        pub(crate) fn as_metal_batch_encode_gemm_f32(
            batch: *mut AsMetalBatch,
            a: *mut AsMetalBuffer,
            b: *mut AsMetalBuffer,
            c: *mut AsMetalBuffer,
            m: u32,
            k: u32,
            n: u32,
        ) -> i32;
        pub(crate) fn as_metal_batch_encode_binary_f32(
            batch: *mut AsMetalBatch,
            op_kind: u32,
            a: *mut AsMetalBuffer,
            b: *mut AsMetalBuffer,
            c: *mut AsMetalBuffer,
            count: u32,
        ) -> i32;
        pub(crate) fn as_metal_batch_encode_relu_f32(
            batch: *mut AsMetalBatch,
            a: *mut AsMetalBuffer,
            c: *mut AsMetalBuffer,
            count: u32,
        ) -> i32;
        pub(crate) fn as_metal_batch_encode_rmsnorm_f32(
            batch: *mut AsMetalBatch,
            x: *mut AsMetalBuffer,
            w: *mut AsMetalBuffer,
            y: *mut AsMetalBuffer,
            rows: u32,
            cols: u32,
            eps: f32,
        ) -> i32;
        pub(crate) fn as_metal_batch_encode_softmax_f32(
            batch: *mut AsMetalBatch,
            x: *mut AsMetalBuffer,
            y: *mut AsMetalBuffer,
            rows: u32,
            cols: u32,
        ) -> i32;
        pub(crate) fn as_metal_batch_commit_and_free(batch: *mut AsMetalBatch) -> i32;
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
        pub(crate) fn as_metal_swiglu_f32(
            ctx: *mut AsMetalContext,
            gate: *const f32,
            up: *const f32,
            out: *mut f32,
            count: u32,
        ) -> i32;
        pub(crate) fn as_metal_rmsnorm_f32(
            ctx: *mut AsMetalContext,
            x: *const f32,
            w: *const f32,
            y: *mut f32,
            rows: u32,
            cols: u32,
            eps: f32,
        ) -> i32;
        pub(crate) fn as_metal_softmax_f32(
            ctx: *mut AsMetalContext,
            x: *const f32,
            y: *mut f32,
            rows: u32,
            cols: u32,
        ) -> i32;
        pub(crate) fn as_metal_rope_f32(
            ctx: *mut AsMetalContext,
            x: *const f32,
            y: *mut f32,
            seq_len: u32,
            head_dim: u32,
            pos_offset: u32,
            theta: f32,
        ) -> i32;
        pub(crate) fn as_metal_q4_matvec_f32(
            ctx: *mut AsMetalContext,
            packed: *const u8,
            scales: *const f32,
            x: *const f32,
            y: *mut f32,
            out_dim: u32,
            in_dim: u32,
            num_blocks_per_row: u32,
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

    /// Allocates a persistent zero-copy Apple Silicon Unified Memory [`MetalBuffer`]
    /// holding `len` `f32` elements (`MTLResourceStorageModeShared`).
    pub fn alloc_f32_buffer(&self, len: usize) -> Result<MetalBuffer> {
        self.limits.check_allocation(len, DType::F32)?;
        if len == 0 {
            return Err(Error::InvalidArgument {
                context: "MetalBackend::alloc_f32_buffer",
                reason: "buffer length must be >= 1".to_owned(),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let byte_len = len.checked_mul(4).ok_or(Error::ArithmeticOverflow {
                context: "MetalBackend::alloc_f32_buffer",
            })?;
            // SAFETY: `self.handle.raw` is a valid live AsMetalContext.
            let raw = unsafe { ffi::as_metal_buf_alloc(self.handle.raw, byte_len) };
            if raw.is_null() {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::alloc_f32_buffer",
                    reason: "Metal buffer allocation failed".to_owned(),
                });
            }
            Ok(MetalBuffer {
                raw,
                len,
                _ctx: Arc::clone(&self.handle),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::alloc_f32_buffer",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Allocates a [`MetalBuffer`] initialized from `data` in Unified Memory.
    pub fn buffer_from_f32(&self, data: &[f32]) -> Result<MetalBuffer> {
        let mut buf = self.alloc_f32_buffer(data.len())?;
        buf.as_f32_slice_mut().copy_from_slice(data);
        Ok(buf)
    }

    /// Begins a batched GPU command buffer ([`MetalCommandBatch`]) for multi-op single submission.
    pub fn begin_batch(&self) -> Result<MetalCommandBatch> {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: `self.handle.raw` is a valid live AsMetalContext.
            let raw = unsafe { ffi::as_metal_batch_begin(self.handle.raw) };
            if raw.is_null() {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::begin_batch",
                    reason: "failed creating Metal command buffer".to_owned(),
                });
            }
            Ok(MetalCommandBatch {
                raw: Some(raw),
                _ctx: Arc::clone(&self.handle),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::begin_batch",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Executes fused SwiGLU (`silu(gate) * up`) directly on the Metal GPU.
    pub fn swiglu_f32(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(Error::BufferLengthMismatch {
                expected: gate.len(),
                actual: up.len(),
            });
        }
        self.limits.check_allocation(gate.len(), DType::F32)?;
        if gate.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(target_os = "macos")]
        {
            let count = u32::try_from(gate.len()).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::swiglu_f32",
            })?;
            let mut out = vec![0.0f32; gate.len()];
            // SAFETY: Slice lengths verified equal to `count`.
            let rc = unsafe {
                ffi::as_metal_swiglu_f32(
                    self.handle.raw,
                    gate.as_ptr(),
                    up.as_ptr(),
                    out.as_mut_ptr(),
                    count,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::swiglu_f32",
                    reason: format!("Metal swiglu kernel failed with status {rc}"),
                });
            }
            Ok(out)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::swiglu_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Executes row-wise RMSNorm `[rows, cols]` with weight `[cols]` directly on the Metal GPU.
    pub fn rmsnorm_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let total = rows.checked_mul(cols).ok_or(Error::ArithmeticOverflow {
            context: "MetalBackend::rmsnorm_f32",
        })?;
        if x.len() != total || weight.len() != cols {
            return Err(Error::BufferLengthMismatch {
                expected: total,
                actual: x.len(),
            });
        }
        self.limits.check_allocation(total, DType::F32)?;
        if total == 0 {
            return Ok(Vec::new());
        }
        #[cfg(target_os = "macos")]
        {
            let r32 = u32::try_from(rows).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::rmsnorm_f32 rows",
            })?;
            let c32 = u32::try_from(cols).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::rmsnorm_f32 cols",
            })?;
            let mut y = vec![0.0f32; total];
            // SAFETY: Slice lengths verified to match rows*cols and cols.
            let rc = unsafe {
                ffi::as_metal_rmsnorm_f32(
                    self.handle.raw,
                    x.as_ptr(),
                    weight.as_ptr(),
                    y.as_mut_ptr(),
                    r32,
                    c32,
                    eps,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::rmsnorm_f32",
                    reason: format!("Metal rmsnorm kernel failed with status {rc}"),
                });
            }
            Ok(y)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::rmsnorm_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Executes numerically stable row-wise Softmax `[rows, cols]` directly on the Metal GPU.
    pub fn softmax_f32(&self, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let total = rows.checked_mul(cols).ok_or(Error::ArithmeticOverflow {
            context: "MetalBackend::softmax_f32",
        })?;
        if x.len() != total {
            return Err(Error::BufferLengthMismatch {
                expected: total,
                actual: x.len(),
            });
        }
        self.limits.check_allocation(total, DType::F32)?;
        if total == 0 {
            return Ok(Vec::new());
        }
        #[cfg(target_os = "macos")]
        {
            let r32 = u32::try_from(rows).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::softmax_f32 rows",
            })?;
            let c32 = u32::try_from(cols).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::softmax_f32 cols",
            })?;
            let mut y = vec![0.0f32; total];
            // SAFETY: Slice lengths verified equal to rows*cols.
            let rc = unsafe {
                ffi::as_metal_softmax_f32(self.handle.raw, x.as_ptr(), y.as_mut_ptr(), r32, c32)
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::softmax_f32",
                    reason: format!("Metal softmax kernel failed with status {rc}"),
                });
            }
            Ok(y)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::softmax_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Executes Rotary Positional Embedding (RoPE) on `[seq_len, head_dim]` directly on the Metal GPU.
    pub fn rope_f32(
        &self,
        x: &[f32],
        seq_len: usize,
        head_dim: usize,
        pos_offset: usize,
        theta: f32,
    ) -> Result<Vec<f32>> {
        let total = seq_len.checked_mul(head_dim).ok_or(Error::ArithmeticOverflow {
            context: "MetalBackend::rope_f32",
        })?;
        if x.len() != total || head_dim < 2 || head_dim % 2 != 0 {
            return Err(Error::InvalidArgument {
                context: "MetalBackend::rope_f32",
                reason: "x length must equal seq_len*head_dim and head_dim must be even >= 2"
                    .to_owned(),
            });
        }
        self.limits.check_allocation(total, DType::F32)?;
        #[cfg(target_os = "macos")]
        {
            let s32 = u32::try_from(seq_len).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::rope_f32 seq_len",
            })?;
            let d32 = u32::try_from(head_dim).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::rope_f32 head_dim",
            })?;
            let p32 = u32::try_from(pos_offset).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::rope_f32 pos_offset",
            })?;
            let mut y = vec![0.0f32; total];
            // SAFETY: Slice lengths verified equal to seq_len*head_dim.
            let rc = unsafe {
                ffi::as_metal_rope_f32(
                    self.handle.raw,
                    x.as_ptr(),
                    y.as_mut_ptr(),
                    s32,
                    d32,
                    p32,
                    theta,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::rope_f32",
                    reason: format!("Metal rope kernel failed with status {rc}"),
                });
            }
            Ok(y)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::rope_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }

    /// Executes 4-bit block-quantized MatVec (group_size=32) directly on the Metal GPU.
    pub fn q4_matvec_f32(
        &self,
        packed: &[u8],
        scales: &[f32],
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let blocks_per_row = in_dim.div_ceil(32);
        let expected_blocks = out_dim
            .checked_mul(blocks_per_row)
            .ok_or(Error::ArithmeticOverflow {
                context: "MetalBackend::q4_matvec_f32 blocks",
            })?;
        let expected_packed = expected_blocks
            .checked_mul(16)
            .ok_or(Error::ArithmeticOverflow {
                context: "MetalBackend::q4_matvec_f32 packed",
            })?;
        if packed.len() != expected_packed || scales.len() != expected_blocks || x.len() != in_dim {
            return Err(Error::BufferLengthMismatch {
                expected: expected_packed,
                actual: packed.len(),
            });
        }
        self.limits.check_allocation(out_dim, DType::F32)?;
        #[cfg(target_os = "macos")]
        {
            let o32 = u32::try_from(out_dim).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::q4_matvec_f32 out_dim",
            })?;
            let i32_u = u32::try_from(in_dim).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::q4_matvec_f32 in_dim",
            })?;
            let b32 = u32::try_from(blocks_per_row).map_err(|_| Error::ArithmeticOverflow {
                context: "MetalBackend::q4_matvec_f32 blocks_per_row",
            })?;
            let mut y = vec![0.0f32; out_dim];
            // SAFETY: All slice lengths validated against out_dim, in_dim, and blocks_per_row above.
            let rc = unsafe {
                ffi::as_metal_q4_matvec_f32(
                    self.handle.raw,
                    packed.as_ptr(),
                    scales.as_ptr(),
                    x.as_ptr(),
                    y.as_mut_ptr(),
                    o32,
                    i32_u,
                    b32,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalBackend::q4_matvec_f32",
                    reason: format!("Metal q4_matvec kernel failed with status {rc}"),
                });
            }
            Ok(y)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalBackend::q4_matvec_f32",
                reason: "Metal unavailable on this OS".to_owned(),
            })
        }
    }
}

/// Zero-copy persistent Apple Silicon Unified Memory GPU buffer (`MTLResourceStorageModeShared`).
pub struct MetalBuffer {
    #[cfg(target_os = "macos")]
    raw: *mut ffi::AsMetalBuffer,
    len: usize,
    #[cfg(target_os = "macos")]
    _ctx: Arc<ffi::MetalHandle>,
}

// SAFETY: `MTLBuffer` with StorageModeShared is safe to send across threads when externally synchronized.
#[cfg(target_os = "macos")]
unsafe impl Send for MetalBuffer {}

impl MetalBuffer {
    /// Number of `f32` elements stored in this Unified Memory buffer.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Zero-copy read view into Unified Memory (valid after GPU batch completes).
    pub fn as_f32_slice(&self) -> &[f32] {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: `self.raw` was allocated for `self.len * 4` bytes and lives as long as `self`.
            let ptr = unsafe { ffi::as_metal_buf_contents(self.raw) } as *const f32;
            unsafe { core::slice::from_raw_parts(ptr, self.len) }
        }
        #[cfg(not(target_os = "macos"))]
        {
            &[]
        }
    }

    /// Zero-copy mutable view into Unified Memory.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: `&mut self` guarantees exclusive access to the underlying Unified Memory buffer.
            let ptr = unsafe { ffi::as_metal_buf_contents(self.raw) } as *mut f32;
            unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
        }
        #[cfg(not(target_os = "macos"))]
        {
            &mut []
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was allocated by `as_metal_buf_alloc` and is freed once.
        unsafe { ffi::as_metal_buf_free(self.raw) };
    }
}

/// Batched GPU command buffer encoding multiple kernels into a single submission.
pub struct MetalCommandBatch {
    #[cfg(target_os = "macos")]
    raw: Option<*mut ffi::AsMetalBatch>,
    #[cfg(target_os = "macos")]
    _ctx: Arc<ffi::MetalHandle>,
}

impl MetalCommandBatch {
    /// Encodes a tiled `f32` GEMM `[m, k] @ [k, n] -> [m, n]` on persistent [`MetalBuffer`]s.
    pub fn encode_gemm_f32(
        &mut self,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(Error::BufferLengthMismatch {
                expected: m * n,
                actual: c.len(),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let batch = self.raw.ok_or(Error::InternalInvariant {
                message: "MetalCommandBatch already committed",
            })?;
            // SAFETY: Buffer lengths validated above; batch is active.
            let rc = unsafe {
                ffi::as_metal_batch_encode_gemm_f32(
                    batch,
                    a.raw,
                    b.raw,
                    c.raw,
                    m as u32,
                    k as u32,
                    n as u32,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalCommandBatch::encode_gemm_f32",
                    reason: format!("encode failed with status {rc}"),
                });
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (a, b, c, m, k, n);
            Err(Error::InvalidArgument {
                context: "MetalCommandBatch::encode_gemm_f32",
                reason: "Metal unavailable".to_owned(),
            })
        }
    }

    /// Encodes an element-wise `f32` add on persistent [`MetalBuffer`]s.
    pub fn encode_add_f32(
        &mut self,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        count: usize,
    ) -> Result<()> {
        self.encode_binary(0, a, b, c, count)
    }

    /// Encodes a fused SwiGLU (`silu(gate) * up`) on persistent [`MetalBuffer`]s.
    pub fn encode_swiglu_f32(
        &mut self,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        out: &MetalBuffer,
        count: usize,
    ) -> Result<()> {
        self.encode_binary(3, gate, up, out, count)
    }

    /// Encodes a ReLU activation on persistent [`MetalBuffer`]s.
    pub fn encode_relu_f32(&mut self, a: &MetalBuffer, c: &MetalBuffer, count: usize) -> Result<()> {
        if a.len() < count || c.len() < count {
            return Err(Error::BufferLengthMismatch {
                expected: count,
                actual: c.len(),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let batch = self.raw.ok_or(Error::InternalInvariant {
                message: "MetalCommandBatch already committed",
            })?;
            // SAFETY: Buffer lengths validated above.
            let rc = unsafe { ffi::as_metal_batch_encode_relu_f32(batch, a.raw, c.raw, count as u32) };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalCommandBatch::encode_relu_f32",
                    reason: format!("encode failed with status {rc}"),
                });
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (a, c, count);
            Err(Error::InvalidArgument {
                context: "MetalCommandBatch::encode_relu_f32",
                reason: "Metal unavailable".to_owned(),
            })
        }
    }

    /// Encodes row-wise RMSNorm `[rows, cols]` on persistent [`MetalBuffer`]s.
    pub fn encode_rmsnorm_f32(
        &mut self,
        x: &MetalBuffer,
        w: &MetalBuffer,
        y: &MetalBuffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        if x.len() < rows * cols || w.len() < cols || y.len() < rows * cols {
            return Err(Error::BufferLengthMismatch {
                expected: rows * cols,
                actual: y.len(),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let batch = self.raw.ok_or(Error::InternalInvariant {
                message: "MetalCommandBatch already committed",
            })?;
            // SAFETY: Buffer lengths validated above.
            let rc = unsafe {
                ffi::as_metal_batch_encode_rmsnorm_f32(
                    batch,
                    x.raw,
                    w.raw,
                    y.raw,
                    rows as u32,
                    cols as u32,
                    eps,
                )
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalCommandBatch::encode_rmsnorm_f32",
                    reason: format!("encode failed with status {rc}"),
                });
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (x, w, y, rows, cols, eps);
            Err(Error::InvalidArgument {
                context: "MetalCommandBatch::encode_rmsnorm_f32",
                reason: "Metal unavailable".to_owned(),
            })
        }
    }

    /// Encodes row-wise Softmax `[rows, cols]` on persistent [`MetalBuffer`]s.
    pub fn encode_softmax_f32(
        &mut self,
        x: &MetalBuffer,
        y: &MetalBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if x.len() < rows * cols || y.len() < rows * cols {
            return Err(Error::BufferLengthMismatch {
                expected: rows * cols,
                actual: y.len(),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let batch = self.raw.ok_or(Error::InternalInvariant {
                message: "MetalCommandBatch already committed",
            })?;
            // SAFETY: Buffer lengths validated above.
            let rc = unsafe {
                ffi::as_metal_batch_encode_softmax_f32(batch, x.raw, y.raw, rows as u32, cols as u32)
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalCommandBatch::encode_softmax_f32",
                    reason: format!("encode failed with status {rc}"),
                });
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (x, y, rows, cols);
            Err(Error::InvalidArgument {
                context: "MetalCommandBatch::encode_softmax_f32",
                reason: "Metal unavailable".to_owned(),
            })
        }
    }

    /// Commits all encoded kernels to the Metal GPU in a single submission and waits for completion.
    pub fn commit_and_wait(mut self) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let batch = self.raw.take().ok_or(Error::InternalInvariant {
                message: "MetalCommandBatch already committed",
            })?;
            // SAFETY: `batch` is owned and consumed exactly once.
            let rc = unsafe { ffi::as_metal_batch_commit_and_free(batch) };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalCommandBatch::commit_and_wait",
                    reason: format!("Metal GPU batch failed with status {rc}"),
                });
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::InvalidArgument {
                context: "MetalCommandBatch::commit_and_wait",
                reason: "Metal unavailable".to_owned(),
            })
        }
    }

    fn encode_binary(
        &mut self,
        op_kind: u32,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        count: usize,
    ) -> Result<()> {
        if a.len() < count || b.len() < count || c.len() < count {
            return Err(Error::BufferLengthMismatch {
                expected: count,
                actual: c.len(),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let batch = self.raw.ok_or(Error::InternalInvariant {
                message: "MetalCommandBatch already committed",
            })?;
            // SAFETY: Buffer lengths validated above.
            let rc = unsafe {
                ffi::as_metal_batch_encode_binary_f32(batch, op_kind, a.raw, b.raw, c.raw, count as u32)
            };
            if rc != 0 {
                return Err(Error::InvalidArgument {
                    context: "MetalCommandBatch::encode_binary",
                    reason: format!("encode failed with status {rc}"),
                });
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (op_kind, a, b, c, count);
            Err(Error::InvalidArgument {
                context: "MetalCommandBatch::encode_binary",
                reason: "Metal unavailable".to_owned(),
            })
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for MetalCommandBatch {
    fn drop(&mut self) {
        if let Some(batch) = self.raw.take() {
            // SAFETY: Ensures no leaked command buffer if dropped without explicit commit.
            let _ = unsafe { ffi::as_metal_batch_commit_and_free(batch) };
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
