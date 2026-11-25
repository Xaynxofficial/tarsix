//! Zero-dependency WebAssembly (`wasm32-unknown-unknown`) ABI bridge for the TypeScript SDK.
//!
//! All `unsafe` pointer dereferencing is strictly confined to this thin FFI boundary,
//! validating non-null pointers and bounds before invoking safe `as_tensor::Tensor` operations.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use as_core::DType;
use as_tensor::Tensor;

struct Registry {
    next_id: u32,
    tensors: HashMap<u32, Tensor>,
}

fn registry() -> &'static Mutex<Registry> {
    static REG: OnceLock<Mutex<Registry>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Registry {
            next_id: 1,
            tensors: HashMap::new(),
        })
    })
}

fn store(t: Tensor) -> u32 {
    let Ok(mut reg) = registry().lock() else {
        return 0;
    };
    let id = reg.next_id;
    reg.next_id = reg.next_id.wrapping_add(1).max(1);
    reg.tensors.insert(id, t);
    id
}

fn fetch(id: u32) -> Option<Tensor> {
    let reg = registry().lock().ok()?;
    reg.tensors.get(&id).cloned()
}

/// Allocates a buffer of `len` `f32` elements in linear memory for host writes.
#[unsafe(no_mangle)]
pub extern "C" fn as_alloc_f32(len: usize) -> *mut f32 {
    let mut v = vec![0.0f32; len];
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

/// Frees a buffer previously allocated with `as_alloc_f32`.
#[unsafe(no_mangle)]
pub extern "C" fn as_free_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        // SAFETY: `ptr` and `len` must match a prior `as_alloc_f32` allocation.
        let _ = unsafe { Vec::from_raw_parts(ptr, len, len) };
    }
}

/// Allocates a buffer of `len` `u32` elements in linear memory for shape descriptors.
#[unsafe(no_mangle)]
pub extern "C" fn as_alloc_u32(len: usize) -> *mut u32 {
    let mut v = vec![0u32; len];
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

/// Frees a buffer previously allocated with `as_alloc_u32`.
#[unsafe(no_mangle)]
pub extern "C" fn as_free_u32(ptr: *mut u32, len: usize) {
    if !ptr.is_null() {
        // SAFETY: `ptr` and `len` must match a prior `as_alloc_u32` allocation.
        let _ = unsafe { Vec::from_raw_parts(ptr, len, len) };
    }
}

fn read_shape(shape_ptr: *const u32, rank: usize) -> Option<Vec<usize>> {
    if rank == 0 {
        return Some(Vec::new());
    }
    if shape_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller passes valid `rank`-element `u32` array in linear memory.
    let slice = unsafe { core::slice::from_raw_parts(shape_ptr, rank) };
    Some(slice.iter().map(|&d| d as usize).collect())
}

/// Creates a zero-filled `F32` tensor. Returns handle `> 0`, or `0` on error.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_zeros(shape_ptr: *const u32, rank: usize) -> u32 {
    let Some(shape) = read_shape(shape_ptr, rank) else {
        return 0;
    };
    Tensor::zeros(shape, DType::F32).map_or(0, store)
}

/// Creates a one-filled `F32` tensor. Returns handle `> 0`, or `0` on error.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_ones(shape_ptr: *const u32, rank: usize) -> u32 {
    let Some(shape) = read_shape(shape_ptr, rank) else {
        return 0;
    };
    Tensor::ones(shape, DType::F32).map_or(0, store)
}

/// Creates an `F32` tensor from linear memory slice and shape.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_from_f32(
    data_ptr: *const f32,
    data_len: usize,
    shape_ptr: *const u32,
    rank: usize,
) -> u32 {
    let Some(shape) = read_shape(shape_ptr, rank) else {
        return 0;
    };
    let data = if data_len == 0 {
        &[]
    } else if data_ptr.is_null() {
        return 0;
    } else {
        // SAFETY: Caller passes valid `data_len`-element `f32` array in linear memory.
        unsafe { core::slice::from_raw_parts(data_ptr, data_len) }
    };
    Tensor::from_f32(data, shape).map_or(0, store)
}

/// Element-wise broadcast addition.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_add(a: u32, b: u32) -> u32 {
    let (Some(ta), Some(tb)) = (fetch(a), fetch(b)) else {
        return 0;
    };
    ta.add(&tb).map_or(0, store)
}

/// Element-wise broadcast subtraction.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_sub(a: u32, b: u32) -> u32 {
    let (Some(ta), Some(tb)) = (fetch(a), fetch(b)) else {
        return 0;
    };
    ta.sub(&tb).map_or(0, store)
}

/// Element-wise broadcast multiplication.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_mul(a: u32, b: u32) -> u32 {
    let (Some(ta), Some(tb)) = (fetch(a), fetch(b)) else {
        return 0;
    };
    ta.mul(&tb).map_or(0, store)
}

/// Element-wise broadcast division.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_div(a: u32, b: u32) -> u32 {
    let (Some(ta), Some(tb)) = (fetch(a), fetch(b)) else {
        return 0;
    };
    ta.div(&tb).map_or(0, store)
}

/// 2D or batched N-D matrix multiplication.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_matmul(a: u32, b: u32) -> u32 {
    let (Some(ta), Some(tb)) = (fetch(a), fetch(b)) else {
        return 0;
    };
    ta.matmul(&tb).map_or(0, store)
}

/// Zero-copy transpose of `axis_a` and `axis_b`.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_transpose(a: u32, axis_a: usize, axis_b: usize) -> u32 {
    let Some(ta) = fetch(a) else {
        return 0;
    };
    ta.transpose(axis_a, axis_b).map_or(0, store)
}

/// Tensor reshape.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_reshape(a: u32, shape_ptr: *const u32, rank: usize) -> u32 {
    let (Some(ta), Some(shape)) = (fetch(a), read_shape(shape_ptr, rank)) else {
        return 0;
    };
    ta.reshape(shape).map_or(0, store)
}

/// Global sum reduction.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_sum(a: u32) -> u32 {
    let Some(ta) = fetch(a) else {
        return 0;
    };
    ta.sum().map_or(0, store)
}

/// Global mean reduction.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_mean(a: u32) -> u32 {
    let Some(ta) = fetch(a) else {
        return 0;
    };
    ta.mean().map_or(0, store)
}

/// Returns total element count for tensor `id`.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_numel(id: u32) -> usize {
    fetch(id).map_or(0, |t| t.numel())
}

/// Copies tensor `id`'s `f32` values into `out_ptr[0..out_len]`. Returns `0` on success.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_read_f32(id: u32, out_ptr: *mut f32, out_len: usize) -> i32 {
    let Some(t) = fetch(id) else {
        return -1;
    };
    let Ok(vals) = t.to_f32_vec() else {
        return -2;
    };
    if vals.len() != out_len || (out_len > 0 && out_ptr.is_null()) {
        return -3;
    }
    if out_len > 0 {
        // SAFETY: Caller allocated writable `out_len` f32 elements at `out_ptr`.
        let dst = unsafe { core::slice::from_raw_parts_mut(out_ptr, out_len) };
        dst.copy_from_slice(&vals);
    }
    0
}

/// Releases tensor `id` from the registry.
#[unsafe(no_mangle)]
pub extern "C" fn as_tensor_dispose(id: u32) {
    if let Ok(mut reg) = registry().lock() {
        reg.tensors.remove(&id);
    }
}
