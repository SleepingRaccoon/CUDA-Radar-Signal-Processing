/*
    cd build && cmake --build . --config Release --target elementwise_add
    cd .. && build\Release\elementwise_add.exe
    build\Release\elementwise_add.exe > easy_examples\elementwise_add\output_elementwise_add.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "elementwise_add.cuh"
#include "helper.cuh"

// relative error verification
// for f32: eps = 1e-5
// for f16: eps = 1e-2 (FP16 precision ~3-4 decimal digits)
static bool verify_f32(const float* gpu, const float* cpu, int n, float eps)
{
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        float maxv = fmaxf(fabsf(gpu[i]), fabsf(cpu[i]));
        float err = (maxv > 1.0f) ? diff / maxv : diff;
        if (err > eps) {
            printf("  FAIL [%d] gpu=%f cpu=%f err=%e\n", i, gpu[i], cpu[i], err);
            return false;
        }
    }
    return true;
}

static bool verify_f16(const half* gpu_half, const float* cpu_f32, int n, float eps)
{
    for (int i = 0; i < n; i++) {
        float gv = __half2float(gpu_half[i]);
        float cv = cpu_f32[i];
        float diff = fabsf(gv - cv);
        float maxv = fmaxf(fabsf(gv), fabsf(cv));
        float err = (maxv > 1.0f) ? diff / maxv : diff;
        if (err > eps) {
            printf("  FAIL [%d] gpu=%f cpu=%f err=%e\n", i, gv, cv, err);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int N = 4 * 1024 * 1024;
    constexpr int REPS = 10;
    constexpr int BLK = 256;

    float *a = new float[N];
    float *b = new float[N];
    float *ref = new float[N];
    half *af = new half[N];
    half *bf = new half[N];
    half *cf = new half[N];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    for (int i = 0; i < N; i++) {
        a[i] = d(rng);
        b[i] = d(rng);
        af[i] = __float2half(a[i]);
        bf[i] = __float2half(b[i]);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) ref[i] = a[i] + b[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *da, *db, *dc;
    half *daf, *dbf, *dcf;
    cudaMalloc(&da, N * sizeof(float));
    cudaMalloc(&db, N * sizeof(float));
    cudaMalloc(&dc, N * sizeof(float));
    cudaMalloc(&daf, N * sizeof(half));
    cudaMalloc(&dbf, N * sizeof(half));
    cudaMalloc(&dcf, N * sizeof(half));
    cudaMemcpy(da, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(daf, af, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dbf, bf, N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 blk(BLK);
    dim3 grd((N + BLK - 1) / BLK);
    dim3 g4((N / 4 + BLK - 1) / BLK);
    dim3 g2((N / 2 + BLK - 1) / BLK);
    dim3 g8((N / 8 + BLK - 1) / BLK);

    double bw32 = 2.0 * N * sizeof(float);
    double bw16 = 2.0 * N * sizeof(half);

    printf("\nelementwise_add  N=%d\n\n", N);
    printf(" Version  Correct   Time(ms)   BW(GB/s)  Eff.BW%%  vsCPU/x\n");
    printf(" -------  -------  ---------  ---------  -------  -------\n");

    auto pr = [&](const char* name, bool ok, float ms, double bw) {
        printf(" %7s  %6s  %9.3f  %9.2f  %7.1f  %7.1f\n",
               name, ok ? "PASS" : "FAIL", ms, bw / (ms/1000) / 1e9,
               bw / (ms/1000) / 1e9 / 192 * 100, cpu_ms / ms);
    };

    float* cg = new float[N];

    // f32
    cudaMemset(dc, 0, N * sizeof(float));
    float ms = KernelTime(elementwise_add_f32, grd, blk, REPS, 0U, nullptr, da, db, dc, N);
    cudaMemcpy(cg, dc, N * sizeof(float), cudaMemcpyDeviceToHost);
    pr("f32", verify_f32(cg, ref, N, 1e-5f), ms, bw32);

    // f32x4
    cudaMemset(dc, 0, N * sizeof(float));
    ms = KernelTime(elementwise_add_f32x4, g4, blk, REPS, 0U, nullptr, da, db, dc, N);
    cudaMemcpy(cg, dc, N * sizeof(float), cudaMemcpyDeviceToHost);
    pr("f32x4", verify_f32(cg, ref, N, 1e-5f), ms, bw32);

    // f16
    cudaMemset(dcf, 0, N * sizeof(half));
    ms = KernelTime(elementwise_add_f16, grd, blk, REPS, 0U, nullptr, daf, dbf, dcf, N);
    cudaMemcpy(cf, dcf, N * sizeof(half), cudaMemcpyDeviceToHost);
    pr("f16", verify_f16(cf, ref, N, 1e-2f), ms, bw16);

    // f16x2
    cudaMemset(dcf, 0, N * sizeof(half));
    ms = time_kernel((void*)elementwise_add_f16x2, g2, blk, N, REPS, daf, dbf, dcf, true);
    cudaMemcpy(cf, dcf, N * sizeof(half), cudaMemcpyDeviceToHost);
    pr("f16x2", verify_f16(cf, ref, N, 1e-2f), ms, bw16);

    // f16x8v1
    cudaMemset(dcf, 0, N * sizeof(half));
    ms = KernelTime(elementwise_add_f16x8_v1, g8, blk, REPS, 0U, nullptr, daf, dbf, dcf, N);
    cudaMemcpy(cf, dcf, N * sizeof(half), cudaMemcpyDeviceToHost);
    pr("f16x8v1", verify_f16(cf, ref, N, 1e-2f), ms, bw16);

    // f16x8v2
    cudaMemset(dcf, 0, N * sizeof(half));
    ms = KernelTime(elementwise_add_f16x8_v2, g8, blk, REPS, 0U, nullptr, daf, dbf, dcf, N);
    cudaMemcpy(cf, dcf, N * sizeof(half), cudaMemcpyDeviceToHost);
    pr("f16x8v2", verify_f16(cf, ref, N, 1e-2f), ms, bw16);

    printf(" -------  -------  ---------  ---------  -------  -------\n");
    printf("     CPU      --  %9.3f  %9.2f      --    1.0 x\n\n",
           cpu_ms, bw32 / (cpu_ms/1000) / 1e9);

    delete[] a; delete[] b; delete[] ref; delete[] cg;
    delete[] af; delete[] bf; delete[] cf;
    cudaFree(da); cudaFree(db); cudaFree(dc);
    cudaFree(daf); cudaFree(dbf); cudaFree(dcf);
    return 0;
}
