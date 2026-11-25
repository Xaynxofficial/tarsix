// Complete runnable C++ example calling the native multi-core Rust Tarsix engine.
// Build & run:
//   cargo build -p as-wasm --release
//   clang++ -std=c++17 examples/cpp_basic.cpp -Iinclude -Ltarget/release -las_wasm -Wl,-rpath,target/release -o /tmp/tarsix_cpp
//   /tmp/tarsix_cpp

#include <cstdio>
#include <vector>
#include "tarsix.h"

int main() {
    uint32_t shape2x2[2] = {2, 2};
    uint32_t shape1x2[2] = {1, 2};

    float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float bias_vals[2] = {1.0f, -2.0f};

    uint32_t a = as_tensor_from_f32(a_vals, 4, shape2x2, 2);
    uint32_t b = as_tensor_from_f32(b_vals, 4, shape2x2, 2);
    uint32_t bias = as_tensor_from_f32(bias_vals, 2, shape1x2, 2);

    uint32_t mm = as_tensor_matmul(a, b);
    uint32_t out = as_tensor_add(mm, bias);

    size_t n = as_tensor_numel(out);
    std::vector<float> result(n);
    if (as_tensor_read_f32(out, result.data(), n) != 0) {
        std::fprintf(stderr, "Failed to read tensor output\n");
        return 1;
    }

    std::printf("C++ -> Native Rust Tarsix result: [%.1f, %.1f, %.1f, %.1f]\n",
                result[0], result[1], result[2], result[3]);

    as_tensor_dispose(a);
    as_tensor_dispose(b);
    as_tensor_dispose(bias);
    as_tensor_dispose(mm);
    as_tensor_dispose(out);
    return 0;
}
