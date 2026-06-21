/*
    cmake --build build --config Release --target reduce
    build\Release\reduce.exe
*/

/**
 * main_reduce.cu
 *
 * Reduce test harness -- compares all kernel variants:
 *   global, shared, shared_stride, syncwrap, shfl, cg
 *
 * NOTE: benchKernel from utils.cuh does NOT reset d_y between
 * repetitions, but every reduce kernel calls atomicAdd on d_y.
 * This file uses a custom bench loop that resets d_y before
 * each run.
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "reduce.cuh"
#include "utils.cuh"

static constexpr int BLK = 256;

struct Result {
    std::string label;
    double      ms;
    double      melem;
    double      cpuSpd;
    bool        pass;
};

// Custom benchmark: reset d_y before each repetition
template <typename KernelFunc, typename... Args>
static double benchReduce(KernelFunc kernel, dim3 grid, dim3 block,
                          int nreps, unsigned int smem,
                          float *d_y, Args... args)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double totalMs = 0.0;
    for (int r = 0; r < nreps; ++r) {
        CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block, smem>>>(args..., d_y);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += static_cast<double>(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return totalMs / nreps;
}

int main()
{
    constexpr int N       = 4 * 1024 * 1024;
    constexpr int NREPS   = 20;
    constexpr double REL_TOL = 1.0e-4;   // 0.01% relative tolerance

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    // ------------------------------------------------------------------
    // 1. Device info
    // ------------------------------------------------------------------
    auto dev = DeviceInfo::query();

    // ------------------------------------------------------------------
    // 2. Host memory + CPU reference
    // ------------------------------------------------------------------
    auto *h_x = new float[N];
    float h_y_cpu = 0.0f, h_y_gpu = 0.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_x[i] = dist(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    reduce_cpu(h_x, N, &h_y_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double cpuMelem = static_cast<double>(N) * 1e-6 / (cpuMs * 1e-3);

    // ------------------------------------------------------------------
    // 3. Device memory
    // ------------------------------------------------------------------
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 4. Launch configs
    // ------------------------------------------------------------------
    dim3 block1(BLK);
    dim3 grid1((N + BLK - 1) / BLK);

    dim3 block2(BLK);
    dim3 grid2(4 * dev.smCount);

    int smemByte = BLK * sizeof(float);

    // ------------------------------------------------------------------
    // 5. Bench all kernels
    // ------------------------------------------------------------------
    std::vector<Result> results;

    auto addResult = [&](const std::string &lbl, double ms, bool ok) {
        double melem = static_cast<double>(N) * 1e-6 / (ms * 1e-3);
        results.push_back({lbl, ms, melem, cpuMs / ms, ok});
    };

    // Warm-up
    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
    reduce_shared<<<grid1, block1, smemByte>>>(d_x, N, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    // -- reduce_global (needs data re-upload per iteration -- it destroys d_x) --
    {
        // Allocate scratch buffer so we don't pollute d_x
        float *d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, bytes));

        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));

        // Warm-up
        CUDA_CHECK(cudaMemcpy(d_tmp, h_x, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
        reduce_global<<<grid1, block1>>>(d_tmp, N, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());

        double totalMs = 0.0;
        for (int r = 0; r < NREPS; ++r) {
            CUDA_CHECK(cudaMemcpy(d_tmp, h_x, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
            CUDA_CHECK(cudaEventRecord(s));
            reduce_global<<<grid1, block1>>>(d_tmp, N, d_y);
            CUDA_CHECK(cudaEventRecord(e));
            CUDA_CHECK(cudaEventSynchronize(e));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
            totalMs += static_cast<double>(ms);
        }

        CUDA_CHECK(cudaEventDestroy(s));
        CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaFree(d_tmp));

        double ms = totalMs / NREPS;
        CUDA_CHECK(cudaMemcpy(&h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(&h_y_gpu, &h_y_cpu, 1, REL_TOL, "global");
        addResult("global", ms, ok);
    }

    // -- reduce_shared --
    {
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        double ms = benchReduce(reduce_shared, grid1, block1, NREPS, smemByte, d_y, d_x, N);
        CUDA_CHECK(cudaMemcpy(&h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(&h_y_gpu, &h_y_cpu, 1, REL_TOL, "shared");
        addResult("shared", ms, ok);
    }

    // -- reduce_shared_stride --
    {
        double ms = benchReduce(reduce_shared_stride, grid2, block2, NREPS, smemByte, d_y, d_x, N);
        CUDA_CHECK(cudaMemcpy(&h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(&h_y_gpu, &h_y_cpu, 1, REL_TOL, "stride");
        addResult("stride", ms, ok);
    }

    // -- reduce_syncwrap --
    {
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        double ms = benchReduce(reduce_syncwrap, grid1, block1, NREPS, smemByte, d_y, d_x, N);
        CUDA_CHECK(cudaMemcpy(&h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(&h_y_gpu, &h_y_cpu, 1, REL_TOL, "syncwrap");
        addResult("syncwrap", ms, ok);
    }

    // -- reduce_shfl --
    {
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        double ms = benchReduce(reduce_shfl, grid1, block1, NREPS, smemByte, d_y, d_x, N);
        CUDA_CHECK(cudaMemcpy(&h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(&h_y_gpu, &h_y_cpu, 1, REL_TOL, "shfl");
        addResult("shfl", ms, ok);
    }

    // -- reduce_cg --
    {
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        double ms = benchReduce(reduce_cg<32>, grid1, block1, NREPS, smemByte, d_y, d_x, N);
        CUDA_CHECK(cudaMemcpy(&h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(&h_y_gpu, &h_y_cpu, 1, REL_TOL, "cg");
        addResult("cg", ms, ok);
    }

    // ==================================================================
    // 6. Print table
    // ==================================================================
    std::cout << "\n\n";
    std::cout << "============================================================\n";
    std::cout << "  reduce -- Performance Summary (" << N << " floats, "
              << (bytes >> 20) << " MB, " << dev.name << ")\n";
    std::cout << "------------------------------------------------------------\n";

    TablePrinter tbl({
        {"Version",   9, 0},
        {"Correct",   7, 0},
        {"Time(ms)", 11, 3},
        {"Melem/s",  11, 1},
        {"vsCPU/x",   9, 1},
    });

    tbl.header();

    for (auto &r : results) {
        tbl.row({
            r.label,
            r.pass ? "PASS" : "FAIL",
            fmt(r.ms,    3, 11),   // 3 decimals, width 11
            fmt(r.melem, 1, 11),
            fmt(r.cpuSpd, 1,  9),
        });
    }

    tbl.sep();
    tbl.row({
        "CPU", "--",
        fmt(cpuMs,    3, 11),
        fmt(cpuMelem, 1, 11),
        "1.0 x",
    });

    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Grid/block:  block=" << grid1.x << "x" << block1.x
              << "  stride=" << grid2.x << "x" << block2.x << "\n";

    bool allPass = true;
    for (auto &r : results) allPass = allPass && r.pass;
    std::cout << "\n  Result: " << (allPass ? "ALL PASSED" : "SOME FAILED") << "\n\n";

    // ==================================================================
    // 7. Write output.txt
    // ==================================================================
    std::string outPath = __FILE__;
    auto pos = outPath.rfind('\\');
    if (pos == std::string::npos) pos = outPath.rfind('/');
    if (pos != std::string::npos) outPath = outPath.substr(0, pos + 1);
    outPath += "reduce_output.txt";

    std::ofstream ofs(outPath);
    if (ofs.is_open()) {
        ofs << "reduce Performance Summary\n=========================\n";
        ofs << "  GPU: " << dev.name << "  |  " << (bytes >> 20) << " MB  |  "
            << N << " floats\n\n";

        tbl.headerTo(ofs);
        for (auto &r : results)
            tbl.rowTo(ofs, {r.label, r.pass ? "PASS" : "FAIL",
                            fmt(r.ms, 3, 11), fmt(r.melem, 1, 11),
                            fmt(r.cpuSpd, 1, 9)});
        tbl.sepTo(ofs);
        tbl.rowTo(ofs, {"CPU", "--", fmt(cpuMs, 3, 11),
                        fmt(cpuMelem, 1, 11), "1.0 x"});
        ofs.close();
    }

    // ------------------------------------------------------------------
    // 8. Cleanup
    // ------------------------------------------------------------------
    delete[] h_x;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return allPass ? EXIT_SUCCESS : EXIT_FAILURE;
}
