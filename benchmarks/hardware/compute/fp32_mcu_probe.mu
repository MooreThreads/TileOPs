// One-shot FP32 throughput probe for MP31 and Moore Perf Compute (mcu).
// The kernel uses independent dependent FMA chains to keep the FP32 pipe busy.

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CHECK_MUSA(call) do { \
    musaError_t err = (call); \
    if (err != musaSuccess) { \
        std::fprintf(stderr, "MUSA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     musaGetErrorString(err)); \
        return 2; \
    } \
} while (0)

__global__ void fp32_fma(float* out, long long iters, float seed) {
    float x0 = seed + 0.1f;
    float x1 = seed + 0.2f;
    float x2 = seed + 0.3f;
    float x3 = seed + 0.4f;
    float x4 = seed + 0.5f;
    float x5 = seed + 0.6f;
    float x6 = seed + 0.7f;
    float x7 = seed + 0.8f;
    const float a = 1.000001f;
    const float b = 0.000001f;
    for (long long i = 0; i < iters; ++i) {
        x0 = fmaf(x0, a, b);
        x1 = fmaf(x1, a, b);
        x2 = fmaf(x2, a, b);
        x3 = fmaf(x3, a, b);
        x4 = fmaf(x4, a, b);
        x5 = fmaf(x5, a, b);
        x6 = fmaf(x6, a, b);
        x7 = fmaf(x7, a, b);
    }
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

template <int ILP>
__global__ void fp32_fma_ilp(float* out, long long iters, float seed) {
    float acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = seed + 0.001f * (float)(i + threadIdx.x);
    const float a = 0.99999994f;
    const float b = 0.0000001f;
    for (long long it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < ILP; ++i) acc[i] = fmaf(acc[i], a, b);
    }
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ILP; ++i) sum += acc[i];
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = sum;
}

template <int ILP>
double event_timed_fma(float* out, int blocks, long long iters, int reps) {
    const int threads = 256;
    const double flops = 2.0 * (double)blocks * threads * ILP * iters;
    for (int i = 0; i < 20; ++i)
        fp32_fma_ilp<ILP><<<blocks, threads>>>(out, iters, 0.73f);
    CHECK_MUSA(musaDeviceSynchronize());

    std::vector<float> run_medians;
    run_medians.reserve(5);
    for (int run = 0; run < 5; ++run) {
        std::vector<float> samples;
        samples.reserve(reps);
        musaEvent_t start, stop;
        CHECK_MUSA(musaEventCreate(&start));
        CHECK_MUSA(musaEventCreate(&stop));
        CHECK_MUSA(musaEventRecord(start));
        for (int i = 0; i < reps; ++i)
            fp32_fma_ilp<ILP><<<blocks, threads>>>(out, iters, 0.73f);
        CHECK_MUSA(musaEventRecord(stop));
        CHECK_MUSA(musaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_MUSA(musaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms / reps);
        CHECK_MUSA(musaEventDestroy(start));
        CHECK_MUSA(musaEventDestroy(stop));
        run_medians.push_back(samples[0]);
    }
    std::sort(run_medians.begin(), run_medians.end());
    const float median_ms = run_medians[run_medians.size() / 2];
    return flops / ((double)median_ms * 1.0e9);
}

int main(int argc, char** argv) {
    const long long iters = argc > 1 ? std::atoll(argv[1]) : 4096;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 480;
    const int ilp = argc > 3 ? std::atoi(argv[3]) : 16;
    const int timed_reps = argc > 4 ? std::atoi(argv[4]) : 0;
    const int threads = 256;
    const long long outputs = (long long)blocks * threads;
    std::printf("probe=fp32_fma gpu=MP31 iters=%lld blocks=%d threads=%d ilp=%d\n",
                iters, blocks, threads, ilp);

    float* out = nullptr;
    CHECK_MUSA(musaMalloc(&out, outputs * sizeof(float)));
    if (ilp == 8) {
        fp32_fma<<<blocks, threads>>>(out, iters, 0.73f);
    } else if (ilp == 16) {
        fp32_fma_ilp<16><<<blocks, threads>>>(out, iters, 0.73f);
    } else if (ilp == 32) {
        fp32_fma_ilp<32><<<blocks, threads>>>(out, iters, 0.73f);
    } else if (ilp == 1) {
        fp32_fma_ilp<1><<<blocks, threads>>>(out, iters, 0.73f);
    } else if (ilp == 2) {
        fp32_fma_ilp<2><<<blocks, threads>>>(out, iters, 0.73f);
    } else if (ilp == 4) {
        fp32_fma_ilp<4><<<blocks, threads>>>(out, iters, 0.73f);
    } else {
        std::fprintf(stderr, "ilp must be 1, 2, 4, 8, 16, or 32\n");
        return 2;
    }
    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());
    if (timed_reps > 0) {
        double tflops = 0.0;
        if (ilp == 1) tflops = event_timed_fma<1>(out, blocks, iters, timed_reps);
        else if (ilp == 2) tflops = event_timed_fma<2>(out, blocks, iters, timed_reps);
        else if (ilp == 4) tflops = event_timed_fma<4>(out, blocks, iters, timed_reps);
        else if (ilp == 8) tflops = event_timed_fma<8>(out, blocks, iters, timed_reps);
        else if (ilp == 16) tflops = event_timed_fma<16>(out, blocks, iters, timed_reps);
        else if (ilp == 32) tflops = event_timed_fma<32>(out, blocks, iters, timed_reps);
        std::printf("event_fp32_fma_tflops=%.6f reps=%d\n", tflops, timed_reps);
    }
    float checksum = 0.0f;
    CHECK_MUSA(musaMemcpy(&checksum, out, sizeof(float), musaMemcpyDeviceToHost));
    std::printf("checksum=%g\n", checksum);
    CHECK_MUSA(musaFree(out));
    return 0;
}
