/*
    cmake --build build --config Release --target sgeva
    build\Release\sgeva.exe
*/

/**
 * main_sgeva.cu
 *
 * sgeva test harness — benchmarks all kernel variants, prints
 * comprehensive metrics in a formatted table.
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "sgeva.cuh"
#include "utils.cuh"

struct Result {
    std::string label;
    double      ms;
    double      bw;       // GB/s
    double      effBW;    // %
    double      gflops;
    double      thru;     // Melem/s
    double      cpuSpd;   // vs CPU
    bool        pass;
};

int main()
{
    constexpr int N       = 4 * 1024 * 1024;
    constexpr int NREPS   = 20;
    constexpr int BLK     = 256;
    constexpr double TOL  = 1e-5f;

    size_t bytes  = static_cast<size_t>(N) * sizeof(float);
    double dataBW = 2.0 * static_cast<double>(bytes);  // read + write

    // ------------------------------------------------------------------
    // 1. Device info
    // ------------------------------------------------------------------
    auto dev = DeviceInfo::query();

    // ------------------------------------------------------------------
    // 2. Host memory + CPU reference
    // ------------------------------------------------------------------
    auto *h_a   = new float[N];
    auto *h_b   = new float[N];
    auto *h_c   = new float[N];
    auto *h_ref = new float[N];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) { h_a[i] = dist(rng); h_b[i] = dist(rng); }

    auto t0 = std::chrono::high_resolution_clock::now();
    sgeva_cpu(h_a, h_b, N, h_ref);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ------------------------------------------------------------------
    // 3. Device memory
    // ------------------------------------------------------------------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 4. Launch configs
    // ------------------------------------------------------------------
    dim3 block(BLK);
    dim3 grid((N + BLK - 1) / BLK);
    dim3 grid4((N / 4 + BLK - 1) / BLK);

    // ------------------------------------------------------------------
    // 5. Bench all kernels
    // ------------------------------------------------------------------
    sgeva_v11<<<grid, block>>>(d_a, d_b, N, d_c);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Result> results;

    auto addResult = [&](const std::string &lbl, double ms,
                          dim3 g, dim3 b, bool ok) {
        double bw   = dataBW / (ms * 1e-3) / 1e9;
        double eff  = 100.0 * bw / dev.bwTheor;
        double thru = static_cast<double>(N) * 1e-6 / (ms * 1e-3);
        double gflops = (2.0 * N) / (ms * 1e-3) / 1e9;
        results.push_back({lbl, ms, bw, eff, gflops, thru, cpuMs / ms, ok});
    };

    // v11: naive scalar
    {
        double ms = benchKernel(sgeva_v11, grid, block, NREPS, 0U, d_a, d_b, N, d_c);
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_c, h_ref, N, 1.0e-6, "v11");
        addResult("v11", ms, grid, block, ok);
    }

    // v12: float4
    {
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        double ms = benchKernel(sgeva_v12, grid4, block, NREPS, 0U, d_a, d_b, N, d_c);
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_c, h_ref, N, 1.0e-6, "v12");
        addResult("v12", ms, grid4, block, ok);
    }

    // v21: grid-stride loop, scalar
    {
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        double ms = benchKernel(sgeva_v21, grid, block, NREPS, 0U, d_a, d_b, N, d_c);
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_c, h_ref, N, 1.0e-6, "v21");
        addResult("v21", ms, grid, block, ok);
    }

    // v22: grid-stride loop + float4
    {
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        double ms = benchKernel(sgeva_v22, grid4, block, NREPS, 0U, d_a, d_b, N, d_c);
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_c, h_ref, N, 1.0e-6, "v22");
        addResult("v22", ms, grid4, block, ok);
    }

    // ==================================================================
    // 6. Print table
    // ==================================================================
    std::cout << "\n\n";
    std::cout << "====================================================================\n";
    std::cout << "  sgeva -- Performance Summary (" << N << " floats, "
              << (bytes >> 20) << " MB, " << dev.name << ")\n";
    std::cout << "--------------------------------------------------------------------\n";

    TablePrinter tbl({
        {"Version",  7, 0},
        {"Correct",  7, 0},
        {"Time(ms)", 9, 3},
        {"BW(GB/s)", 9, 2},
        {"Eff.BW%",  8, 1},
        {"GFLOPS",   8, 1},
        {"Melem/s",  9, 1},
        {"vsCPU/x",  8, 1},
    });

    tbl.header();

    for (size_t i = 0; i < results.size(); ++i) {
        auto &r = results[i];
        if (i == 2) tbl.sep();   // separate naive vs grid-stride groups

        int occ = computeOccPct(reinterpret_cast<const void *>(sgeva_v11),
                                 (i == 0 || i == 2) ? BLK : BLK, dev);

        tbl.row({
            r.label,
            r.pass ? "PASS" : "FAIL",
            fmt(r.ms,      3, 9),
            fmt(r.bw,      2, 9),
            fmt(r.effBW,   1, 8),
            std::to_string(occ),
            fmt(r.gflops,  1, 8),
            fmt(r.thru,    1, 9),
            fmt(r.cpuSpd,  1, 8),
        });
    }

    // CPU row
    double cpuBW  = dataBW / (cpuMs * 1e-3) / 1e9;
    double cpuThr = static_cast<double>(N) * 1e-6 / (cpuMs * 1e-3);
    tbl.sep();
    tbl.row({
        "CPU", "--",
        fmt(cpuMs, 3, 9),
        fmt(cpuBW,  2, 9),
        "--", "0.0",
        fmt(cpuThr, 1, 9),
        "1.0 x",
    });

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "  Theoretical BW: " << std::fixed << std::setprecision(2)
              << dev.bwTheor << " GB/s  |  "
              << static_cast<int>(results[0].effBW) << "% (v11)  "
              << static_cast<int>(results[1].effBW) << "% (v12)\n";
    std::cout << "  v12/v11 speedup: " << std::fixed << std::setprecision(2)
              << results[0].ms / results[1].ms << "x  |  "
              << "  v22/v21 speedup: "
              << results[2].ms / results[3].ms << "x\n";

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
    outPath += "output_sgeva.txt";

    std::ofstream ofs(outPath);
    if (ofs.is_open()) {
        ofs << "sgeva Performance Summary\n=========================\n";
        ofs << "  GPU: " << dev.name << "  |  "
            << (bytes >> 20) << " MB  |  "
            << N << " floats  |  Theoretical BW: "
            << dev.bwTheor << " GB/s\n\n";

        tbl.headerTo(ofs);
        for (size_t i = 0; i < results.size(); ++i) {
            auto &r = results[i];
            if (i == 2) tbl.sepTo(ofs);
            tbl.rowTo(ofs, {
                r.label, r.pass ? "PASS" : "FAIL",
                fmt(r.ms, 3, 9), fmt(r.bw, 2, 9),
                fmt(r.effBW, 1, 8), fmt(r.gflops, 1, 8),
                fmt(r.thru, 1, 9), fmt(r.cpuSpd, 1, 8),
            });
        }
        tbl.sepTo(ofs);
        tbl.rowTo(ofs, {
            "CPU", "--", fmt(cpuMs, 3, 9), fmt(cpuBW, 2, 9),
            "--", "0.0", fmt(cpuThr, 1, 9), "1.0 x",
        });
        ofs.close();
    }

    // ------------------------------------------------------------------
    // 8. Cleanup
    // ------------------------------------------------------------------
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));

    return allPass ? EXIT_SUCCESS : EXIT_FAILURE;
}
