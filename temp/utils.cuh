#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__          \
                      << "  " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ===========================================================================
// Benchmark kernel (template)
// ===========================================================================
template <typename KernelFunc, typename... Args>
static double benchKernel(KernelFunc kernel, dim3 grid, dim3 block,
                          int nreps, unsigned int smem, Args... args)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    kernel<<<grid, block, smem>>>(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    double totalMs = 0.0;
    for (int r = 0; r < nreps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(args...);
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

// ===========================================================================
// Verification
// ===========================================================================
static bool verify(const float *gpu, const float *cpu, int n,
                   double relTol, const std::string &label)
{
    double maxRelErr = 0.0;
    int    firstBad  = -1;
    for (int i = 0; i < n; ++i) {
        double ref  = static_cast<double>(cpu[i]);
        double val  = static_cast<double>(gpu[i]);
        double diff = ref - val;
        if (diff < 0.0) diff = -diff;
        double relErr = (ref != 0.0) ? diff / ref : diff;
        if (relErr > maxRelErr) maxRelErr = relErr;
        if (relErr > relTol && firstBad < 0) firstBad = i;
    }

    if (firstBad >= 0) {
        std::string tag = "[" + std::to_string(firstBad) + "]";
        std::cout << std::left << std::setw(10) << label
                  << " FAIL  " << tag
                  << " GPU=" << gpu[firstBad]
                  << " CPU=" << cpu[firstBad] << std::endl;
        return false;
    }
    std::cout << std::left << std::setw(10) << label
              << " PASS  (maxRelErr=" << std::scientific
              << maxRelErr << std::fixed << ")" << std::endl;
    return true;
}

// ===========================================================================
// Device info helper
// ===========================================================================
struct DeviceInfo {
    std::string name;
    double      bwTheor;       // GB/s
    int         smCount;
    int         warpSize;
    int         maxThreadsPerSM;

    static DeviceInfo query(int device = 0) {
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        int memClock_kHz = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClock_kHz,
                                           cudaDevAttrMemoryClockRate, device));
        double bw = memClock_kHz * 1e3
                    * static_cast<double>(prop.memoryBusWidth) / 8.0 * 2.0 / 1e9;

        return {
            prop.name,
            bw,
            prop.multiProcessorCount,
            prop.warpSize,
            prop.maxThreadsPerMultiProcessor
        };
    }
};

// ===========================================================================
// Column-based table printer
// ===========================================================================
struct TableCol {
    std::string header;
    int         width;
    int         decimals;   // fixed-point decimals, 0 = plain int
};

class TablePrinter {
    std::vector<TableCol> cols_;

public:
    explicit TablePrinter(std::vector<TableCol> cols) : cols_(std::move(cols)) {}

    // Print header row + separator
    void header() const {
        for (auto &c : cols_) std::cout << " " << std::right << std::setw(c.width) << c.header;
        std::cout << '\n';
        sep();
    }

    void sep() const {
        for (auto &c : cols_) std::cout << " " << std::string(c.width, '-');
        std::cout << '\n';
    }

    // Print one row from string values
    void row(const std::vector<std::string> &vals) const {
        for (size_t i = 0; i < vals.size(); ++i)
            std::cout << " " << std::right << std::setw(cols_[i].width) << vals[i];
        std::cout << '\n';
    }

    // Same to ofstream
    void rowTo(std::ofstream &os, const std::vector<std::string> &vals) const {
        for (size_t i = 0; i < vals.size(); ++i)
            os << " " << std::right << std::setw(cols_[i].width) << vals[i];
        os << '\n';
    }

    void sepTo(std::ofstream &os) const {
        for (auto &c : cols_) os << " " << std::string(c.width, '-');
        os << '\n';
    }

    void headerTo(std::ofstream &os) const {
        for (auto &c : cols_) os << " " << std::right << std::setw(c.width) << c.header;
        os << '\n';
        sepTo(os);
    }

    const std::vector<TableCol> &cols() const { return cols_; }

    size_t colCount() const { return cols_.size(); }
};

// Format a fixed-point number into a string, respects column width
inline std::string fmt(double v, int decimals, int width) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << v;
    std::string s = oss.str();
    if (static_cast<int>(s.size()) > width) s = std::string(width, '#');
    return s;
}

// ===========================================================================
// Theoretical occupancy for a given block size
// ===========================================================================
inline int computeOccPct(const void *kernel, int blockSize,
                         const DeviceInfo &dev)
{
    int numBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, kernel, blockSize, 0);
    int warpsPerSM = numBlocks * (blockSize / dev.warpSize);
    int maxWarps   = dev.maxThreadsPerSM / dev.warpSize;
    return static_cast<int>(100.0 * warpsPerSM / maxWarps);
}
