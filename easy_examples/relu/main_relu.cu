/*
    cd build && cmake --build . --config Release --target relu
    cd .. && build\Release\relu.exe
    build\Release\relu.exe > easy_examples\relu\output_relu.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "relu.cuh"
#include "helper.cuh"

int main()
{
    constexpr int N = 4 * 1024 * 1024;
    constexpr int REPS = 10;
    constexpr int BLK = 256;

    float *xf = new float[N];
    float *yf = new float[N];
    float *ref = new float[N];
    half *xh = new half[N];
    half *yh = new half[N];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    for (int i = 0; i < N; i++) {
        xf[i] = d(rng);
        xh[i] = __float2half(xf[i]);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    relu_f32_cpu(xf, ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dx, *dy;
    half *dxh, *dyh;
    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, N * sizeof(float));
    cudaMalloc(&dxh, N * sizeof(half));
    cudaMalloc(&dyh, N * sizeof(half));
    cudaMemcpy(dx, xf, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dxh, xh, N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 blk(BLK);
    dim3 grd((N + BLK - 1) / BLK);
    dim3 g4((N / 4 + BLK - 1) / BLK);
    dim3 g2((N / 2 + BLK - 1) / BLK);
    dim3 g8((N / 8 + BLK - 1) / BLK);

    double bw32 = 2.0 * N * sizeof(float);
    double bw16 = 2.0 * N * sizeof(half);

    auto verify_f32 = [&]() {
        cudaMemcpy(yf, dy, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++)
            if (fabsf(yf[i] - ref[i]) > 1e-5f) { printf("  FAIL [%d]\n", i); return false; }
        return true;
    };

    auto verify_f16 = [&]() {
        cudaMemcpy(yh, dyh, N * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++)
            if (fabsf(__half2float(yh[i]) - ref[i]) > 1e-2f) { printf("  FAIL [%d]\n", i); return false; }
        return true;
    };

    printf("\nrelu  N=%d\n\n", N);
    printf(" Version       Correct     Time(ms)     BW(GB/s)    Melem/s    vsCPU/x\n");
    printf(" ---------    --------    ---------    ---------    -------    -------\n");

    // warmup
    relu_f32<<<grd, blk>>>(dx, dy, N);
    cudaDeviceSynchronize();

    float ms, bw, thru, spd;
    bool ok;

    // f32
    cudaMemset(dy, 0, N * sizeof(float));
    relu_f32<<<grd, blk>>>(dx, dy, N); cudaDeviceSynchronize();
    ok = verify_f32();
    ms = KernelTime(relu_f32, grd, blk, REPS, 0U, nullptr, dx, dy, N);
    bw = bw32 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("     f32         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f32x4
    cudaMemset(dy, 0, N * sizeof(float));
    relu_f32x4<<<g4, blk>>>(dx, dy, N); cudaDeviceSynchronize();
    ok = verify_f32();
    ms = KernelTime(relu_f32x4, g4, blk, REPS, 0U, nullptr, dx, dy, N);
    bw = bw32 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("   f32x4         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16
    cudaMemset(dyh, 0, N * sizeof(half));
    relu_f16<<<grd, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(relu_f16, grd, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("     f16         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16x2
    cudaMemset(dyh, 0, N * sizeof(half));
    relu_f16x2<<<g2, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(relu_f16x2, g2, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("   f16x2         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16x8v1
    cudaMemset(dyh, 0, N * sizeof(half));
    relu_f16x8_v1<<<g8, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(relu_f16x8_v1, g8, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf(" f16x8v1         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16x8v2
    cudaMemset(dyh, 0, N * sizeof(half));
    relu_f16x8_v2<<<g8, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(relu_f16x8_v2, g8, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf(" f16x8v2         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    printf(" ---------    --------    ---------    ---------    -------    -------\n");
    printf("     CPU         --        %7.3f      %7.2f      %7.1f      1.0 x\n\n",
           cpu_ms, bw32 / (cpu_ms / 1000) / 1e9, (double)N / 1e6 / (cpu_ms / 1000));

    delete[] xf; delete[] yf; delete[] ref; delete[] xh; delete[] yh;
    cudaFree(dx); cudaFree(dy); cudaFree(dxh); cudaFree(dyh);
    return 0;
}
