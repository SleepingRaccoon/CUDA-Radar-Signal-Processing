/*
    cd build && cmake --build . --config Release --target histogram
    cd .. && build\Release\histogram.exe
    build\Release\histogram.exe > easy_examples\histogram\output_histogram.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

#include "histogram.cuh"
#include "helper.cuh"

int main()
{
    constexpr int N = 4 * 1024 * 1024;
    constexpr int BINS = 1024;
    constexpr int REPS = 20;
    constexpr int BLK = 256;

    int *a = new int[N];
    int *ref = new int[BINS]();
    int *cg = new int[BINS]();

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(0, BINS - 1);
    for (int i = 0; i < N; i++) a[i] = d(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    histogram_cpu(a, ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int *da, *dy;
    cudaMalloc(&da, N * sizeof(int));
    cudaMalloc(&dy, BINS * sizeof(int));
    cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blk(BLK);
    dim3 grd((N + BLK - 1) / BLK);
    dim3 g4((N / 4 + BLK - 1) / BLK);

    auto verify = [&]() {
        cudaMemcpy(cg, dy, BINS * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < BINS; i++) {
            float diff = (float)abs(cg[i] - ref[i]);
            float maxv = fmaxf((float)cg[i], (float)ref[i]);
            float err = (maxv > 1.0f) ? diff / maxv : diff;
            if (err > 1e-5f) {
                printf("  FAIL [%d] gpu=%d cpu=%d err=%e\n", i, cg[i], ref[i], err);
                return false;
            }
        }
        return true;
    };

    // warmup
    histogram_i32<<<grd, blk>>>(da, dy, N);
    cudaDeviceSynchronize();

    printf("\nhistogram  N=%d  BINS=%d\n\n", N, BINS);
    printf(" Version       Correct     Time(ms)     Melem/s    vsCPU/x\n");
    printf(" ---------    --------    ---------    -------    -------\n");

    // i32
    cudaMemset(dy, 0, BINS * sizeof(int));
    histogram_i32<<<grd, blk>>>(da, dy, N); cudaDeviceSynchronize();
    bool ok = verify();
    float ms = KernelTime(histogram_i32, grd, blk, REPS, 0U, nullptr, da, dy, N);
    printf("     i32         %s       %7.3f      %7.1f      %7.1f\n",
           ok ? "PASS" : "FAIL", ms, (double)N / 1e6 / (ms / 1000), cpu_ms / ms);

    // i32x4
    cudaMemset(dy, 0, BINS * sizeof(int));
    histogram_i32x4<<<g4, blk>>>(da, dy, N); cudaDeviceSynchronize();
    ok = verify();
    ms = KernelTime(histogram_i32x4, g4, blk, REPS, 0U, nullptr, da, dy, N);
    printf("   i32x4         %s       %7.3f      %7.1f      %7.1f\n",
           ok ? "PASS" : "FAIL", ms, (double)N / 1e6 / (ms / 1000), cpu_ms / ms);

    printf(" ---------    --------    ---------    -------    -------\n");
    printf("     CPU         --        %7.3f      %7.1f      1.0 x\n\n",
           cpu_ms, (double)N / 1e6 / (cpu_ms / 1000));

    delete[] a; delete[] ref; delete[] cg;
    cudaFree(da); cudaFree(dy);
    return 0;
}
