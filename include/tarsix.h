/*
 * Tarsix — Independent AI Computing Infrastructure (C / C++ ABI Header)
 * License: Apache-2.0
 *
 * Link against `libas_wasm.dylib` (macOS), `libas_wasm.so` (Linux), or `as_wasm.dll` (Windows)
 * built via: `cargo build -p as-wasm --release`
 */

#ifndef TARSIX_H
#define TARSIX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Linear memory helpers (primarily for WASM / FFI callers) */
float*    as_alloc_f32(size_t len);
void      as_free_f32(float* ptr, size_t len);
uint32_t* as_alloc_u32(size_t len);
void      as_free_u32(uint32_t* ptr, size_t len);

/* Tensor constructors (return handle > 0 on success, 0 on error) */
uint32_t  as_tensor_zeros(const uint32_t* shape_ptr, size_t rank);
uint32_t  as_tensor_ones(const uint32_t* shape_ptr, size_t rank);
uint32_t  as_tensor_from_f32(
    const float* data_ptr,
    size_t data_len,
    const uint32_t* shape_ptr,
    size_t rank
);

/* Broadcast-aware arithmetic & linear algebra */
uint32_t  as_tensor_add(uint32_t a, uint32_t b);
uint32_t  as_tensor_sub(uint32_t a, uint32_t b);
uint32_t  as_tensor_mul(uint32_t a, uint32_t b);
uint32_t  as_tensor_div(uint32_t a, uint32_t b);
uint32_t  as_tensor_matmul(uint32_t a, uint32_t b);

/* Zero-copy view & reduction operations */
uint32_t  as_tensor_transpose(uint32_t a, size_t axis_a, size_t axis_b);
uint32_t  as_tensor_reshape(uint32_t a, const uint32_t* shape_ptr, size_t rank);
uint32_t  as_tensor_sum(uint32_t a);
uint32_t  as_tensor_mean(uint32_t a);

/* Readback & lifecycle */
size_t    as_tensor_numel(uint32_t id);
int32_t   as_tensor_read_f32(uint32_t id, float* out_ptr, size_t out_len);
void      as_tensor_dispose(uint32_t id);

#ifdef __cplusplus
}
#endif

#endif /* TARSIX_H */
