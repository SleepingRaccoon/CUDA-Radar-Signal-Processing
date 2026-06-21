/*
    cd build && cmake --build . --config Release --target sigmoid
    cd .. && build\Release\sigmoid.exe
    build\Release\sigmoid.exe > easy_examples\sigmoid\output_sigmoid.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "sigmoid.cuh"
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
    sigmoid_f32_cpu(xf, ref, N);
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

    double bw_f32 = 2.0 * N * sizeof(float);
    double bw_f16 = 2.0 * N * sizeof(half);

    auto verify_f32 = [&]() {
        cudaMemcpy(yf, dy, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            float err = fabsf(yf[i] - ref[i]);
            if (err > 1e-4f) { printf("  FAIL [%d] gpu=%f cpu=%f\n", i, yf[i], ref[i]); return false; }
        }
        return true;
    };

    auto verify_f16 = [&]() {
        cudaMemcpy(yh, dyh, N * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            float gv = __half2float(yh[i]), cv = ref[i];
            float err = fabsf(gv - cv);
            if (err > 1e-2f) { printf("  FAIL [%d] gpu=%f cpu=%f\n", i, gv, cv); return false; }
        }
        return true;
    };

    printf("\nsigmoid  N=%d\n\n", N);
    printf(" Version       Correct     Time(ms)     BW(GB/s)    Melem/s    vsCPU/x\n");
    printf(" ---------    --------    ---------    ---------    -------    -------\n");

    float ms, bw, thru, spd;

    // warmup
    sigmoid_f32<<<grd, blk>>>(dx, dy, N);
    cudaDeviceSynchronize();

    // f32
    cudaMemset(dy, 0, N * sizeof(float));
    sigmoid_f32<<<grd, blk>>>(dx, dy, N); cudaDeviceSynchronize();
    bool ok = verify_f32();
    ms = KernelTime(sigmoid_f32, grd, blk, REPS, 0U, nullptr, dx, dy, N);
    bw = bw_f32 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("     f32         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f32x4
    cudaMemset(dy, 0, N * sizeof(float));
    sigmoid_f32x4<<<g4, blk>>>(dx, dy, N); cudaDeviceSynchronize();
    ok = verify_f32();
    ms = KernelTime(sigmoid_f32x4, g4, blk, REPS, 0U, nullptr, dx, dy, N);
    bw = bw_f32 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("   f32x4         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16
    cudaMemset(dyh, 0, N * sizeof(half));
    sigmoid_f16<<<grd, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(sigmoid_f16, grd, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw_f16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("     f16         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16x2
    cudaMemset(dyh, 0, N * sizeof(half));
    sigmoid_f16x2<<<g2, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(sigmoid_f16x2, g2, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw_f16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf("   f16x2         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16x8v1
    cudaMemset(dyh, 0, N * sizeof(half));
    sigmoid_f16x8_v1<<<g8, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(sigmoid_f16x8_v1, g8, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw_f16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf(" f16x8v1         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    // f16x8v2
    cudaMemset(dyh, 0, N * sizeof(half));
    sigmoid_f16x8_v2<<<g8, blk>>>(dxh, dyh, N); cudaDeviceSynchronize();
    ok = verify_f16();
    ms = KernelTime(sigmoid_f16x8_v2, g8, blk, REPS, 0U, nullptr, dxh, dyh, N);
    bw = bw_f16 / (ms / 1000) / 1e9; thru = (double)N / 1e6 / (ms / 1000); spd = cpu_ms / ms;
    printf(" f16x8v2         %s       %7.3f      %7.2f      %7.1f      %7.1f\n", ok ? "PASS" : "FAIL", ms, bw, thru, spd);

    printf(" ---------    --------    ---------    ---------    -------    -------\n");
    printf("     CPU         --        %7.3f      %7.2f      %7.1f      1.0 x\n\n",
           cpu_ms, bw_f32 / (cpu_ms / 1000) / 1e9, (double)N / 1e6 / (cpu_ms / 1000));

    delete[] xf; delete[] yf; delete[] ref; delete[] xh; delete[] yh;
    cudaFree(dx); cudaFree(dy); cudaFree(dxh); cudaFree(dyh);
    return 0;
}
