// Audited Objective-C bridge to Apple Metal.framework (zero external crates).
// Supports:
// - Zero-copy Apple Silicon Unified Memory buffers (MTLResourceStorageModeShared)
// - Batched MTLCommandBuffer submission (multi-op single-wait execution)
// - Production LLM & NN kernels: Tiled GEMM, Add/Sub/Mul/ReLU, RMSNorm, Softmax, SwiGLU, RoPE, Q4 MatVec
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t M;
    uint32_t K;
    uint32_t N;
} GemmDims;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} NormParams;

typedef struct {
    uint32_t seq_len;
    uint32_t head_dim;
    uint32_t pos_offset;
    float theta;
} RopeParams;

typedef struct {
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t num_blocks_per_row;
} Q4MatVecParams;

typedef struct {
    void* device;
    void* queue;
    void* gemm_pipeline;
    void* add_pipeline;
    void* sub_pipeline;
    void* mul_pipeline;
    void* relu_pipeline;
    void* rmsnorm_pipeline;
    void* softmax_pipeline;
    void* swiglu_pipeline;
    void* rope_pipeline;
    void* q4_matvec_pipeline;
} AsMetalContext;

typedef struct {
    void* buffer;
    size_t byte_len;
} AsMetalBuffer;

typedef struct {
    AsMetalContext* ctx;
    void* cmd_buf;
} AsMetalBatch;

static const char* kAsMetalShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct GemmDims { uint M; uint K; uint N; };\n"
"struct NormParams { uint rows; uint cols; float eps; };\n"
"struct RopeParams { uint seq_len; uint head_dim; uint pos_offset; float theta; };\n"
"struct Q4MatVecParams { uint out_dim; uint in_dim; uint num_blocks_per_row; };\n"
"\n"
"kernel void gemm_f32_tiled(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C [[buffer(2)]],\n"
"    constant GemmDims& dims [[buffer(3)]],\n"
"    uint2 gid [[thread_position_in_grid]],\n"
"    uint2 tid [[thread_position_in_threadgroup]]\n"
") {\n"
"    constexpr uint TILE = 16;\n"
"    threadgroup float As[TILE][TILE];\n"
"    threadgroup float Bs[TILE][TILE];\n"
"    uint row = gid.y;\n"
"    uint col = gid.x;\n"
"    float acc = 0.0f;\n"
"    uint num_tiles = (dims.K + TILE - 1) / TILE;\n"
"    for (uint t = 0; t < num_tiles; ++t) {\n"
"        uint a_col = t * TILE + tid.x;\n"
"        uint b_row = t * TILE + tid.y;\n"
"        As[tid.y][tid.x] = (row < dims.M && a_col < dims.K) ? A[row * dims.K + a_col] : 0.0f;\n"
"        Bs[tid.y][tid.x] = (b_row < dims.K && col < dims.N) ? B[b_row * dims.N + col] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint k = 0; k < TILE; ++k) {\n"
"            acc = fma(As[tid.y][k], Bs[k][tid.x], acc);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < dims.M && col < dims.N) {\n"
"        C[row * dims.N + col] = acc;\n"
"    }\n"
"}\n"
"\n"
"kernel void add_f32(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C [[buffer(2)]],\n"
"    constant uint& count [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid < count) { C[gid] = A[gid] + B[gid]; }\n"
"}\n"
"\n"
"kernel void sub_f32(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C [[buffer(2)]],\n"
"    constant uint& count [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid < count) { C[gid] = A[gid] - B[gid]; }\n"
"}\n"
"\n"
"kernel void mul_f32(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C [[buffer(2)]],\n"
"    constant uint& count [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid < count) { C[gid] = A[gid] * B[gid]; }\n"
"}\n"
"\n"
"kernel void relu_f32(\n"
"    device const float* A [[buffer(0)]],\n"
"    device float* C [[buffer(1)]],\n"
"    constant uint& count [[buffer(2)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid < count) { float v = A[gid]; C[gid] = v > 0.0f ? v : 0.0f; }\n"
"}\n"
"\n"
"kernel void swiglu_f32(\n"
"    device const float* gate [[buffer(0)]],\n"
"    device const float* up [[buffer(1)]],\n"
"    device float* out [[buffer(2)]],\n"
"    constant uint& count [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid < count) {\n"
"        float g = gate[gid];\n"
"        float silu = g / (1.0f + exp(-g));\n"
"        out[gid] = silu * up[gid];\n"
"    }\n"
"}\n"
"\n"
"kernel void rmsnorm_f32(\n"
"    device const float* X [[buffer(0)]],\n"
"    device const float* W [[buffer(1)]],\n"
"    device float* Y [[buffer(2)]],\n"
"    constant NormParams& p [[buffer(3)]],\n"
"    uint row [[thread_position_in_grid]]\n"
") {\n"
"    if (row >= p.rows) return;\n"
"    uint base = row * p.cols;\n"
"    float sum_sq = 0.0f;\n"
"    for (uint c = 0; c < p.cols; ++c) {\n"
"        float v = X[base + c];\n"
"        sum_sq = fma(v, v, sum_sq);\n"
"    }\n"
"    float inv_rms = rsqrt(sum_sq / float(p.cols) + p.eps);\n"
"    for (uint c = 0; c < p.cols; ++c) {\n"
"        Y[base + c] = X[base + c] * inv_rms * W[c];\n"
"    }\n"
"}\n"
"\n"
"kernel void softmax_f32(\n"
"    device const float* X [[buffer(0)]],\n"
"    device float* Y [[buffer(1)]],\n"
"    constant NormParams& p [[buffer(2)]],\n"
"    uint row [[thread_position_in_grid]]\n"
") {\n"
"    if (row >= p.rows) return;\n"
"    uint base = row * p.cols;\n"
"    float max_v = X[base];\n"
"    for (uint c = 1; c < p.cols; ++c) {\n"
"        max_v = max(max_v, X[base + c]);\n"
"    }\n"
"    float sum_exp = 0.0f;\n"
"    for (uint c = 0; c < p.cols; ++c) {\n"
"        float e = exp(X[base + c] - max_v);\n"
"        Y[base + c] = e;\n"
"        sum_exp += e;\n"
"    }\n"
"    float inv_sum = 1.0f / sum_exp;\n"
"    for (uint c = 0; c < p.cols; ++c) {\n"
"        Y[base + c] *= inv_sum;\n"
"    }\n"
"}\n"
"\n"
"kernel void rope_f32(\n"
"    device const float* X [[buffer(0)]],\n"
"    device float* Y [[buffer(1)]],\n"
"    constant RopeParams& p [[buffer(2)]],\n"
"    uint2 gid [[thread_position_in_grid]]\n"
") {\n"
"    uint half_d = p.head_dim / 2;\n"
"    uint i = gid.x;\n"
"    uint s = gid.y;\n"
"    if (s >= p.seq_len || i >= half_d) return;\n"
"    float pos = float(p.pos_offset + s);\n"
"    float freq = pow(p.theta, -2.0f * float(i) / float(p.head_dim));\n"
"    float angle = pos * freq;\n"
"    float c = cos(angle);\n"
"    float sn = sin(angle);\n"
"    uint base = s * p.head_dim;\n"
"    float x0 = X[base + i];\n"
"    float x1 = X[base + i + half_d];\n"
"    Y[base + i] = x0 * c - x1 * sn;\n"
"    Y[base + i + half_d] = x0 * sn + x1 * c;\n"
"}\n"
"\n"
"// 4-bit block-quantized MatVec (group_size = 32, 16 packed bytes + 1 f32 scale per block)\n"
"kernel void q4_matvec_f32(\n"
"    device const uchar* packed [[buffer(0)]],\n"
"    device const float* scales [[buffer(1)]],\n"
"    device const float* x [[buffer(2)]],\n"
"    device float* y [[buffer(3)]],\n"
"    constant Q4MatVecParams& p [[buffer(4)]],\n"
"    uint row [[thread_position_in_grid]]\n"
") {\n"
"    if (row >= p.out_dim) return;\n"
"    float acc = 0.0f;\n"
"    uint row_blk_base = row * p.num_blocks_per_row;\n"
"    for (uint b = 0; b < p.num_blocks_per_row; ++b) {\n"
"        uint blk_idx = row_blk_base + b;\n"
"        float scale = scales[blk_idx];\n"
"        uint byte_base = blk_idx * 16;\n"
"        uint col_base = b * 32;\n"
"        float blk_acc = 0.0f;\n"
"        for (uint k = 0; k < 16; ++k) {\n"
"            uchar packed_byte = packed[byte_base + k];\n"
"            float q0 = float(int(packed_byte & 0x0F) - 8);\n"
"            float q1 = float(int((packed_byte >> 4) & 0x0F) - 8);\n"
"            uint c0 = col_base + k * 2;\n"
"            uint c1 = c0 + 1;\n"
"            if (c0 < p.in_dim) { blk_acc = fma(q0, x[c0], blk_acc); }\n"
"            if (c1 < p.in_dim) { blk_acc = fma(q1, x[c1], blk_acc); }\n"
"        }\n"
"        acc = fma(blk_acc, scale, acc);\n"
"    }\n"
"    y[row] = acc;\n"
"}\n";

static id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> lib,
    NSString* name
) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [device newComputePipelineStateWithFunction:fn error:&err];
}

AsMetalContext* as_metal_create(char* name_buf, size_t name_buf_len) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;
        if (name_buf && name_buf_len > 0) {
            const char* utf8 = [[device name] UTF8String];
            if (utf8) {
                strncpy(name_buf, utf8, name_buf_len - 1);
                name_buf[name_buf_len - 1] = '\0';
            }
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return NULL;
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kAsMetalShaderSource];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) return NULL;

        id<MTLComputePipelineState> gemm = make_pipeline(device, lib, @"gemm_f32_tiled");
        id<MTLComputePipelineState> add = make_pipeline(device, lib, @"add_f32");
        id<MTLComputePipelineState> sub = make_pipeline(device, lib, @"sub_f32");
        id<MTLComputePipelineState> mul = make_pipeline(device, lib, @"mul_f32");
        id<MTLComputePipelineState> relu = make_pipeline(device, lib, @"relu_f32");
        id<MTLComputePipelineState> rmsnorm = make_pipeline(device, lib, @"rmsnorm_f32");
        id<MTLComputePipelineState> softmax = make_pipeline(device, lib, @"softmax_f32");
        id<MTLComputePipelineState> swiglu = make_pipeline(device, lib, @"swiglu_f32");
        id<MTLComputePipelineState> rope = make_pipeline(device, lib, @"rope_f32");
        id<MTLComputePipelineState> q4mv = make_pipeline(device, lib, @"q4_matvec_f32");

        if (!gemm || !add || !sub || !mul || !relu || !rmsnorm || !softmax || !swiglu || !rope || !q4mv) {
            return NULL;
        }

        AsMetalContext* ctx = (AsMetalContext*)calloc(1, sizeof(AsMetalContext));
        if (!ctx) return NULL;
        ctx->device = (__bridge_retained void*)device;
        ctx->queue = (__bridge_retained void*)queue;
        ctx->gemm_pipeline = (__bridge_retained void*)gemm;
        ctx->add_pipeline = (__bridge_retained void*)add;
        ctx->sub_pipeline = (__bridge_retained void*)sub;
        ctx->mul_pipeline = (__bridge_retained void*)mul;
        ctx->relu_pipeline = (__bridge_retained void*)relu;
        ctx->rmsnorm_pipeline = (__bridge_retained void*)rmsnorm;
        ctx->softmax_pipeline = (__bridge_retained void*)softmax;
        ctx->swiglu_pipeline = (__bridge_retained void*)swiglu;
        ctx->rope_pipeline = (__bridge_retained void*)rope;
        ctx->q4_matvec_pipeline = (__bridge_retained void*)q4mv;
        return ctx;
    }
}

void as_metal_destroy(AsMetalContext* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        if (ctx->q4_matvec_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->q4_matvec_pipeline; }
        if (ctx->rope_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->rope_pipeline; }
        if (ctx->swiglu_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->swiglu_pipeline; }
        if (ctx->softmax_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->softmax_pipeline; }
        if (ctx->rmsnorm_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->rmsnorm_pipeline; }
        if (ctx->relu_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->relu_pipeline; }
        if (ctx->mul_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->mul_pipeline; }
        if (ctx->sub_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->sub_pipeline; }
        if (ctx->add_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->add_pipeline; }
        if (ctx->gemm_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->gemm_pipeline; }
        if (ctx->queue) { (void)(__bridge_transfer id<MTLCommandQueue>)ctx->queue; }
        if (ctx->device) { (void)(__bridge_transfer id<MTLDevice>)ctx->device; }
        free(ctx);
    }
}

// --- Persistent Zero-Copy Unified Memory Buffers ---

AsMetalBuffer* as_metal_buf_alloc(AsMetalContext* ctx, size_t byte_len) {
    if (!ctx || byte_len == 0) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLBuffer> buf = [device newBufferWithLength:byte_len options:MTLResourceStorageModeShared];
        if (!buf) return NULL;
        AsMetalBuffer* out = (AsMetalBuffer*)calloc(1, sizeof(AsMetalBuffer));
        if (!out) return NULL;
        out->buffer = (__bridge_retained void*)buf;
        out->byte_len = byte_len;
        return out;
    }
}

void* as_metal_buf_contents(AsMetalBuffer* buf) {
    if (!buf || !buf->buffer) return NULL;
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf->buffer;
    return [mtl_buf contents];
}

void as_metal_buf_free(AsMetalBuffer* buf) {
    if (!buf) return;
    @autoreleasepool {
        if (buf->buffer) {
            (void)(__bridge_transfer id<MTLBuffer>)buf->buffer;
        }
        free(buf);
    }
}

// --- Batched Command Buffer API (Multi-Op Single Submission) ---

AsMetalBatch* as_metal_batch_begin(AsMetalContext* ctx) {
    if (!ctx) return NULL;
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) return NULL;
        AsMetalBatch* batch = (AsMetalBatch*)calloc(1, sizeof(AsMetalBatch));
        if (!batch) return NULL;
        batch->ctx = ctx;
        batch->cmd_buf = (__bridge_retained void*)cmd;
        return batch;
    }
}

int32_t as_metal_batch_encode_gemm_f32(
    AsMetalBatch* batch,
    AsMetalBuffer* a,
    AsMetalBuffer* b,
    AsMetalBuffer* c,
    uint32_t m,
    uint32_t k,
    uint32_t n
) {
    if (!batch || !a || !b || !c || m == 0 || k == 0 || n == 0) return -1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)batch->cmd_buf;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)batch->ctx->gemm_pipeline;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -2;
        GemmDims dims = { m, k, n };
        [enc setComputePipelineState:pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)a->buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->buffer offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)c->buffer offset:0 atIndex:2];
        [enc setBytes:&dims length:sizeof(GemmDims) atIndex:3];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((n + 15) / 16) * 16, ((m + 15) / 16) * 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        return 0;
    }
}

int32_t as_metal_batch_encode_binary_f32(
    AsMetalBatch* batch,
    uint32_t op_kind, // 0=add, 1=sub, 2=mul, 3=swiglu
    AsMetalBuffer* a,
    AsMetalBuffer* b,
    AsMetalBuffer* c,
    uint32_t count
) {
    if (!batch || !a || !b || !c || count == 0) return -1;
    @autoreleasepool {
        void* raw_pipe = NULL;
        switch (op_kind) {
            case 0: raw_pipe = batch->ctx->add_pipeline; break;
            case 1: raw_pipe = batch->ctx->sub_pipeline; break;
            case 2: raw_pipe = batch->ctx->mul_pipeline; break;
            case 3: raw_pipe = batch->ctx->swiglu_pipeline; break;
            default: return -1;
        }
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)batch->cmd_buf;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)raw_pipe;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -2;
        [enc setComputePipelineState:pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)a->buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)b->buffer offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)c->buffer offset:0 atIndex:2];
        [enc setBytes:&count length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
        if (tg_size > 256) tg_size = 256;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        return 0;
    }
}

int32_t as_metal_batch_encode_relu_f32(
    AsMetalBatch* batch,
    AsMetalBuffer* a,
    AsMetalBuffer* c,
    uint32_t count
) {
    if (!batch || !a || !c || count == 0) return -1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)batch->cmd_buf;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)batch->ctx->relu_pipeline;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -2;
        [enc setComputePipelineState:pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)a->buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)c->buffer offset:0 atIndex:1];
        [enc setBytes:&count length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
        if (tg_size > 256) tg_size = 256;
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        return 0;
    }
}

int32_t as_metal_batch_encode_rmsnorm_f32(
    AsMetalBatch* batch,
    AsMetalBuffer* x,
    AsMetalBuffer* w,
    AsMetalBuffer* y,
    uint32_t rows,
    uint32_t cols,
    float eps
) {
    if (!batch || !x || !w || !y || rows == 0 || cols == 0) return -1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)batch->cmd_buf;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)batch->ctx->rmsnorm_pipeline;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -2;
        NormParams p = { rows, cols, eps };
        [enc setComputePipelineState:pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)x->buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)w->buffer offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)y->buffer offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(NormParams) atIndex:3];
        NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
        if (tg_size > 64) tg_size = 64;
        if (tg_size > rows) tg_size = rows;
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        return 0;
    }
}

int32_t as_metal_batch_encode_softmax_f32(
    AsMetalBatch* batch,
    AsMetalBuffer* x,
    AsMetalBuffer* y,
    uint32_t rows,
    uint32_t cols
) {
    if (!batch || !x || !y || rows == 0 || cols == 0) return -1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)batch->cmd_buf;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)batch->ctx->softmax_pipeline;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -2;
        NormParams p = { rows, cols, 0.0f };
        [enc setComputePipelineState:pipe];
        [enc setBuffer:(__bridge id<MTLBuffer>)x->buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)y->buffer offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(NormParams) atIndex:2];
        NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
        if (tg_size > 64) tg_size = 64;
        if (tg_size > rows) tg_size = rows;
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        return 0;
    }
}

int32_t as_metal_batch_commit_and_free(AsMetalBatch* batch) {
    if (!batch) return -1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)batch->cmd_buf;
        free(batch);
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -3;
    }
}

// --- Direct Single-Shot Convenience Entrypoints ---

int32_t as_metal_gemm_f32(
    AsMetalContext* ctx, const float* a, const float* b, float* c, uint32_t m, uint32_t k, uint32_t n
) {
    if (!ctx || !a || !b || !c || m == 0 || k == 0 || n == 0) return -1;
    AsMetalBuffer* ba = as_metal_buf_alloc(ctx, (size_t)m * k * sizeof(float));
    AsMetalBuffer* bb = as_metal_buf_alloc(ctx, (size_t)k * n * sizeof(float));
    AsMetalBuffer* bc = as_metal_buf_alloc(ctx, (size_t)m * n * sizeof(float));
    if (!ba || !bb || !bc) {
        as_metal_buf_free(ba); as_metal_buf_free(bb); as_metal_buf_free(bc);
        return -2;
    }
    memcpy(as_metal_buf_contents(ba), a, ba->byte_len);
    memcpy(as_metal_buf_contents(bb), b, bb->byte_len);
    AsMetalBatch* batch = as_metal_batch_begin(ctx);
    int32_t rc = as_metal_batch_encode_gemm_f32(batch, ba, bb, bc, m, k, n);
    if (rc == 0) rc = as_metal_batch_commit_and_free(batch);
    if (rc == 0) memcpy(c, as_metal_buf_contents(bc), bc->byte_len);
    as_metal_buf_free(ba); as_metal_buf_free(bb); as_metal_buf_free(bc);
    return rc;
}

int32_t as_metal_add_f32(AsMetalContext* ctx, const float* a, const float* b, float* c, uint32_t count) {
    if (!ctx || !a || !b || !c || count == 0) return -1;
    size_t bytes = (size_t)count * sizeof(float);
    AsMetalBuffer* ba = as_metal_buf_alloc(ctx, bytes);
    AsMetalBuffer* bb = as_metal_buf_alloc(ctx, bytes);
    AsMetalBuffer* bc = as_metal_buf_alloc(ctx, bytes);
    if (!ba || !bb || !bc) {
        as_metal_buf_free(ba); as_metal_buf_free(bb); as_metal_buf_free(bc);
        return -2;
    }
    memcpy(as_metal_buf_contents(ba), a, bytes);
    memcpy(as_metal_buf_contents(bb), b, bytes);
    AsMetalBatch* batch = as_metal_batch_begin(ctx);
    int32_t rc = as_metal_batch_encode_binary_f32(batch, 0, ba, bb, bc, count);
    if (rc == 0) rc = as_metal_batch_commit_and_free(batch);
    if (rc == 0) memcpy(c, as_metal_buf_contents(bc), bytes);
    as_metal_buf_free(ba); as_metal_buf_free(bb); as_metal_buf_free(bc);
    return rc;
}

int32_t as_metal_swiglu_f32(AsMetalContext* ctx, const float* gate, const float* up, float* out, uint32_t count) {
    if (!ctx || !gate || !up || !out || count == 0) return -1;
    size_t bytes = (size_t)count * sizeof(float);
    AsMetalBuffer* bg = as_metal_buf_alloc(ctx, bytes);
    AsMetalBuffer* bu = as_metal_buf_alloc(ctx, bytes);
    AsMetalBuffer* bo = as_metal_buf_alloc(ctx, bytes);
    if (!bg || !bu || !bo) {
        as_metal_buf_free(bg); as_metal_buf_free(bu); as_metal_buf_free(bo);
        return -2;
    }
    memcpy(as_metal_buf_contents(bg), gate, bytes);
    memcpy(as_metal_buf_contents(bu), up, bytes);
    AsMetalBatch* batch = as_metal_batch_begin(ctx);
    int32_t rc = as_metal_batch_encode_binary_f32(batch, 3, bg, bu, bo, count);
    if (rc == 0) rc = as_metal_batch_commit_and_free(batch);
    if (rc == 0) memcpy(out, as_metal_buf_contents(bo), bytes);
    as_metal_buf_free(bg); as_metal_buf_free(bu); as_metal_buf_free(bo);
    return rc;
}

int32_t as_metal_rmsnorm_f32(
    AsMetalContext* ctx, const float* x, const float* w, float* y, uint32_t rows, uint32_t cols, float eps
) {
    if (!ctx || !x || !w || !y || rows == 0 || cols == 0) return -1;
    AsMetalBuffer* bx = as_metal_buf_alloc(ctx, (size_t)rows * cols * sizeof(float));
    AsMetalBuffer* bw = as_metal_buf_alloc(ctx, (size_t)cols * sizeof(float));
    AsMetalBuffer* by = as_metal_buf_alloc(ctx, (size_t)rows * cols * sizeof(float));
    if (!bx || !bw || !by) {
        as_metal_buf_free(bx); as_metal_buf_free(bw); as_metal_buf_free(by);
        return -2;
    }
    memcpy(as_metal_buf_contents(bx), x, bx->byte_len);
    memcpy(as_metal_buf_contents(bw), w, bw->byte_len);
    AsMetalBatch* batch = as_metal_batch_begin(ctx);
    int32_t rc = as_metal_batch_encode_rmsnorm_f32(batch, bx, bw, by, rows, cols, eps);
    if (rc == 0) rc = as_metal_batch_commit_and_free(batch);
    if (rc == 0) memcpy(y, as_metal_buf_contents(by), by->byte_len);
    as_metal_buf_free(bx); as_metal_buf_free(bw); as_metal_buf_free(by);
    return rc;
}

int32_t as_metal_softmax_f32(
    AsMetalContext* ctx, const float* x, float* y, uint32_t rows, uint32_t cols
) {
    if (!ctx || !x || !y || rows == 0 || cols == 0) return -1;
    AsMetalBuffer* bx = as_metal_buf_alloc(ctx, (size_t)rows * cols * sizeof(float));
    AsMetalBuffer* by = as_metal_buf_alloc(ctx, (size_t)rows * cols * sizeof(float));
    if (!bx || !by) {
        as_metal_buf_free(bx); as_metal_buf_free(by);
        return -2;
    }
    memcpy(as_metal_buf_contents(bx), x, bx->byte_len);
    AsMetalBatch* batch = as_metal_batch_begin(ctx);
    int32_t rc = as_metal_batch_encode_softmax_f32(batch, bx, by, rows, cols);
    if (rc == 0) rc = as_metal_batch_commit_and_free(batch);
    if (rc == 0) memcpy(y, as_metal_buf_contents(by), by->byte_len);
    as_metal_buf_free(bx); as_metal_buf_free(by);
    return rc;
}

int32_t as_metal_rope_f32(
    AsMetalContext* ctx, const float* x, float* y, uint32_t seq_len, uint32_t head_dim, uint32_t pos_offset, float theta
) {
    if (!ctx || !x || !y || seq_len == 0 || head_dim < 2 || (head_dim % 2) != 0) return -1;
    @autoreleasepool {
        size_t bytes = (size_t)seq_len * head_dim * sizeof(float);
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)ctx->rope_pipeline;
        id<MTLBuffer> bx = [device newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bx || !by) return -2;

        RopeParams p = { seq_len, head_dim, pos_offset, theta };
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:by offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(RopeParams) atIndex:2];
        uint32_t half_d = head_dim / 2;
        MTLSize tg = MTLSizeMake(half_d < 16 ? half_d : 16, seq_len < 16 ? seq_len : 16, 1);
        MTLSize grid = MTLSizeMake(half_d, seq_len, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) return -3;
        memcpy(y, [by contents], bytes);
        return 0;
    }
}

int32_t as_metal_q4_matvec_f32(
    AsMetalContext* ctx,
    const uint8_t* packed,
    const float* scales,
    const float* x,
    float* y,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t num_blocks_per_row
) {
    if (!ctx || !packed || !scales || !x || !y || out_dim == 0 || in_dim == 0 || num_blocks_per_row == 0) {
        return -1;
    }
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)ctx->q4_matvec_pipeline;

        size_t packed_bytes = (size_t)out_dim * num_blocks_per_row * 16;
        size_t scales_bytes = (size_t)out_dim * num_blocks_per_row * sizeof(float);
        size_t x_bytes = (size_t)in_dim * sizeof(float);
        size_t y_bytes = (size_t)out_dim * sizeof(float);

        id<MTLBuffer> bp = [device newBufferWithBytes:packed length:packed_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bs = [device newBufferWithBytes:scales length:scales_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
        if (!bp || !bs || !bx || !by) return -2;

        Q4MatVecParams p = { out_dim, in_dim, num_blocks_per_row };
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:bp offset:0 atIndex:0];
        [enc setBuffer:bs offset:0 atIndex:1];
        [enc setBuffer:bx offset:0 atIndex:2];
        [enc setBuffer:by offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(Q4MatVecParams) atIndex:4];
        NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
        if (tg_size > 64) tg_size = 64;
        if (tg_size > out_dim) tg_size = out_dim;
        [enc dispatchThreads:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) return -3;
        memcpy(y, [by contents], y_bytes);
        return 0;
    }
}
