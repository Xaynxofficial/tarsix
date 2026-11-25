// Minimal, audited Objective-C bridge to Apple Metal.framework (zero external crates).
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
    void* device;
    void* queue;
    void* gemm_pipeline;
    void* add_pipeline;
} AsMetalContext;

static const char* kAsMetalShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct GemmDims { uint M; uint K; uint N; };\n"
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
"            acc += As[tid.y][k] * Bs[k][tid.x];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < dims.M && col < dims.N) {\n"
"        C[row * dims.N + col] = acc;\n"
"    }\n"
"}\n"
"kernel void add_f32(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C [[buffer(2)]],\n"
"    constant uint& count [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid < count) {\n"
"        C[gid] = A[gid] + B[gid];\n"
"    }\n"
"}\n";

AsMetalContext* as_metal_create(char* name_buf, size_t name_buf_len) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return NULL;
        }
        if (name_buf && name_buf_len > 0) {
            const char* utf8 = [[device name] UTF8String];
            if (utf8) {
                strncpy(name_buf, utf8, name_buf_len - 1);
                name_buf[name_buf_len - 1] = '\0';
            }
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            return NULL;
        }
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kAsMetalShaderSource];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            return NULL;
        }
        id<MTLFunction> gemm_fn = [lib newFunctionWithName:@"gemm_f32_tiled"];
        id<MTLFunction> add_fn = [lib newFunctionWithName:@"add_f32"];
        if (!gemm_fn || !add_fn) {
            return NULL;
        }
        id<MTLComputePipelineState> gemm_pipe =
            [device newComputePipelineStateWithFunction:gemm_fn error:&err];
        id<MTLComputePipelineState> add_pipe =
            [device newComputePipelineStateWithFunction:add_fn error:&err];
        if (!gemm_pipe || !add_pipe) {
            return NULL;
        }

        AsMetalContext* ctx = (AsMetalContext*)calloc(1, sizeof(AsMetalContext));
        if (!ctx) {
            return NULL;
        }
        ctx->device = (__bridge_retained void*)device;
        ctx->queue = (__bridge_retained void*)queue;
        ctx->gemm_pipeline = (__bridge_retained void*)gemm_pipe;
        ctx->add_pipeline = (__bridge_retained void*)add_pipe;
        return ctx;
    }
}

void as_metal_destroy(AsMetalContext* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        if (ctx->add_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->add_pipeline; }
        if (ctx->gemm_pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)ctx->gemm_pipeline; }
        if (ctx->queue) { (void)(__bridge_transfer id<MTLCommandQueue>)ctx->queue; }
        if (ctx->device) { (void)(__bridge_transfer id<MTLDevice>)ctx->device; }
        free(ctx);
    }
}

int32_t as_metal_gemm_f32(
    AsMetalContext* ctx,
    const float* a,
    const float* b,
    float* c,
    uint32_t m,
    uint32_t k,
    uint32_t n
) {
    if (!ctx || !a || !b || !c || m == 0 || k == 0 || n == 0) {
        return -1;
    }
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)ctx->gemm_pipeline;

        size_t a_bytes = (size_t)m * (size_t)k * sizeof(float);
        size_t b_bytes = (size_t)k * (size_t)n * sizeof(float);
        size_t c_bytes = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> buf_a = [device newBufferWithBytes:a length:a_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [device newBufferWithBytes:b length:b_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_c = [device newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];
        if (!buf_a || !buf_b || !buf_c) {
            return -2;
        }

        GemmDims dims = { m, k, n };
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        [enc setBytes:&dims length:sizeof(GemmDims) atIndex:3];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((n + 15) / 16) * 16, ((m + 15) / 16) * 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            return -3;
        }
        memcpy(c, [buf_c contents], c_bytes);
        return 0;
    }
}

int32_t as_metal_add_f32(
    AsMetalContext* ctx,
    const float* a,
    const float* b,
    float* c,
    uint32_t count
) {
    if (!ctx || !a || !b || !c || count == 0) {
        return -1;
    }
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)ctx->add_pipeline;

        size_t bytes = (size_t)count * sizeof(float);
        id<MTLBuffer> buf_a = [device newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [device newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_c = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf_a || !buf_b || !buf_c) {
            return -2;
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        [enc setBytes:&count length:sizeof(uint32_t) atIndex:3];

        NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
        if (tg_size > 256) tg_size = 256;
        MTLSize tg = MTLSizeMake(tg_size, 1, 1);
        MTLSize grid = MTLSizeMake(count, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            return -3;
        }
        memcpy(c, [buf_c contents], bytes);
        return 0;
    }
}
