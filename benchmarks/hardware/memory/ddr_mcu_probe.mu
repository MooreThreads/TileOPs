// One-shot MP31 DDR probes for Moore Perf Compute (mcu).
//
// This source intentionally launches exactly one selected kernel.  It is a
// counter-collection target, not a latency benchmark: use MUSA events without
// mcu for steady-state timing and use mcu for physical DDR counters.

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK_MUSA(call) do { \
    musaError_t err = (call); \
    if (err != musaSuccess) { \
        std::fprintf(stderr, "MUSA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     musaGetErrorString(err)); \
        return 2; \
    } \
} while (0)

__global__ void ddr_copy(const float4* __restrict__ src,
                         float4* __restrict__ dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void ddr_triad(const float4* __restrict__ b,
                          const float4* __restrict__ c,
                          float4* __restrict__ a, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float4 vb = b[i];
        float4 vc = c[i];
        a[i] = make_float4(vb.x + 3.14f * vc.x, vb.y + 3.14f * vc.y,
                           vb.z + 3.14f * vc.z, vb.w + 3.14f * vc.w);
    }
}

__global__ void ddr_read(const float4* __restrict__ src,
                         volatile float* __restrict__ sink, long long n) {
    float sum = 0.0f;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x % 32 == 0) atomicAdd((float*)sink, sum);
}

__global__ void ddr_write(float4* __restrict__ dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float4 value = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    for (; i < n; i += stride) dst[i] = value;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "read";
    long long size_mb = argc > 2 ? std::atoll(argv[2]) : 512;
    const long long bytes = size_mb * 1024LL * 1024LL;
    const long long n = bytes / sizeof(float4);

    musaDeviceProp prop{};
    CHECK_MUSA(musaGetDeviceProperties(&prop, 0));
    const int blocks = prop.multiProcessorCount * 8;
    std::printf("probe=%s gpu=%s sms=%d bytes=%lld blocks=%d\n",
                mode, prop.name, prop.multiProcessorCount, bytes, blocks);

    float4 *src = nullptr, *src2 = nullptr, *dst = nullptr;
    float* sink = nullptr;
    CHECK_MUSA(musaMalloc(&src, bytes));
    CHECK_MUSA(musaMalloc(&src2, bytes));
    CHECK_MUSA(musaMalloc(&dst, bytes));
    CHECK_MUSA(musaMalloc(&sink, sizeof(float)));
    CHECK_MUSA(musaMemset(src, 0x3f, bytes));
    CHECK_MUSA(musaMemset(src2, 0x40, bytes));
    CHECK_MUSA(musaMemset(dst, 0, bytes));
    CHECK_MUSA(musaMemset(sink, 0, sizeof(float)));

    if (std::string(mode) == "copy") {
        ddr_copy<<<blocks, 256>>>(src, dst, n);
    } else if (std::string(mode) == "triad") {
        ddr_triad<<<blocks, 256>>>(src, src2, dst, n);
    } else if (std::string(mode) == "read") {
        ddr_read<<<blocks, 256>>>(src, sink, n);
    } else if (std::string(mode) == "write") {
        ddr_write<<<blocks, 256>>>(dst, n);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }
    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());
    CHECK_MUSA(musaFree(src));
    CHECK_MUSA(musaFree(src2));
    CHECK_MUSA(musaFree(dst));
    CHECK_MUSA(musaFree(sink));
    return 0;
}
