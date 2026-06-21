/**
 * transpose_float4.cu
 *
 * Matrix transpose using float4 (128-bit) vectorized memory accesses.
 *
 * Each thread loads/stores 4 consecutive floats as a single float4,
 * achieving higher effective memory bandwidth than scalar versions.
 *
 * Block configuration: TILE_SZ x (TILE_SZ/4) = 32x8 = 256 threads.
 * Each thread handles 4 float elements.
 *
 * Versions:
 *   v1 - shared-memory tile, bank conflict present
 *   v2 - shared-memory tile + 1-column padding (no bank conflict)
 *   v3 - shared-memory tile + XOR indexing (no bank conflict)
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// Error-check macro
// --------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__              \
                  << "  " << cudaGetErrorString(err) << std::endl;          \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

// Reinterpret a float lvalue through which to load 128 bits as float4.
// Usage: float4 tmp = LD_F4(A[row * w + col]);
#define LD_F4(src)  (*reinterpret_cast<const float4 *>(&(src)))

// Reinterpret a float lvalue through which to store 128 bits.
// Usage: ST_F4(B[row_out * h + col_out]) = tmp;
#define ST_F4(dst)  (*reinterpret_cast<float4 *>(&(dst)))

// ===========================================================================
// CPU reference
// ===========================================================================
static void transpose_cpu(const float *A, int h, int w, float *B)
{
    for (int y = 0; y < w; ++y)
        for (int x = 0; x < h; ++x)
            B[y * h + x] = A[x * w + y];
}

// ===========================================================================
// v1: shared-memory tile, float4 vectorized, bank conflict present
// ===========================================================================
//   blockDim = (TILE_SZ, TILE_SZ/4).
//   Each thread loads one float4 (4 consecutive floats from a row) and
//   scatters them across 4 tile columns. Read-back transposes through
//   shared memory, reading 4 consecutive floats from a column.
// ===========================================================================
template <int TILE_SZ>
__global__ void transpose_float4_v1(const float *A, int h, int w, float *B)
{
    __shared__ float tile[TILE_SZ][TILE_SZ];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int sx  = tid % (TILE_SZ / 4);   // float4 group index: 0..TILE_SZ/4-1
    int sy  = tid / (TILE_SZ / 4);   // row within tile: 0..TILE_SZ-1

    // ---- Load: coalesced float4 read from A ----
    int row = blockIdx.y * TILE_SZ + sy;
    int col = blockIdx.x * TILE_SZ + 4 * sx;

    if (row < h && col < w) {
        float4 tmp = LD_F4(A[row * w + col]);
        tile[sy][4 * sx + 0] = tmp.x;
        tile[sy][4 * sx + 1] = tmp.y;
        tile[sy][4 * sx + 2] = tmp.z;
        tile[sy][4 * sx + 3] = tmp.w;
    }

    __syncthreads();

    // ---- Write-back: transpose via shared memory, coalesced float4 write ----
    int row_out = blockIdx.x * TILE_SZ + sy;       // swapped blockIdx
    int col_out = blockIdx.y * TILE_SZ + 4 * sx;   // swapped blockIdx

    if (row_out < w && col_out < h) {
        float4 tmp;
        tmp.x = tile[4 * sx + 0][sy];
        tmp.y = tile[4 * sx + 1][sy];
        tmp.z = tile[4 * sx + 2][sy];
        tmp.w = tile[4 * sx + 3][sy];
        ST_F4(B[row_out * h + col_out]) = tmp;
    }
}

// ===========================================================================
// v2: shared-memory tile + padding, no bank conflict
// ===========================================================================
template <int TILE_SZ>
__global__ void transpose_float4_v2(const float *A, int h, int w, float *B)
{
    __shared__ float tile[TILE_SZ][TILE_SZ + 1];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int sx  = tid % (TILE_SZ / 4);
    int sy  = tid / (TILE_SZ / 4);

    // ---- Load ----
    int row = blockIdx.y * TILE_SZ + sy;
    int col = blockIdx.x * TILE_SZ + 4 * sx;

    if (row < h && col < w) {
        float4 tmp = LD_F4(A[row * w + col]);
        tile[sy][4 * sx + 0] = tmp.x;
        tile[sy][4 * sx + 1] = tmp.y;
        tile[sy][4 * sx + 2] = tmp.z;
        tile[sy][4 * sx + 3] = tmp.w;
    }

    __syncthreads();

    // ---- Write-back ----
    int row_out = blockIdx.x * TILE_SZ + sy;
    int col_out = blockIdx.y * TILE_SZ + 4 * sx;

    if (row_out < w && col_out < h) {
        float4 tmp;
        tmp.x = tile[4 * sx + 0][sy];
        tmp.y = tile[4 * sx + 1][sy];
        tmp.z = tile[4 * sx + 2][sy];
        tmp.w = tile[4 * sx + 3][sy];
        ST_F4(B[row_out * h + col_out]) = tmp;
    }
}

// ===========================================================================
// v3: shared-memory tile + XOR indexing, no bank conflict
// ===========================================================================
template <int TILE_SZ>
__global__ void transpose_float4_v3(const float *A, int h, int w, float *B)
{
    __shared__ float tile[TILE_SZ][TILE_SZ];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int sx  = tid % (TILE_SZ / 4);
    int sy  = tid / (TILE_SZ / 4);

    // ---- Load: XOR column index to avoid bank conflict ----
    //   Store at tile[row][row ^ col] to scatter consecutive accesses
    //   across different banks.
    int row = blockIdx.y * TILE_SZ + sy;
    int col = blockIdx.x * TILE_SZ + 4 * sx;

    if (row < h && col < w) {
        float4 tmp = LD_F4(A[row * w + col]);
        tile[sy][sy ^ (4 * sx + 0)] = tmp.x;
        tile[sy][sy ^ (4 * sx + 1)] = tmp.y;
        tile[sy][sy ^ (4 * sx + 2)] = tmp.z;
        tile[sy][sy ^ (4 * sx + 3)] = tmp.w;
    }

    __syncthreads();

    // ---- Write-back: XOR read ----
    //   Concept position tile[4*sx+k][sy] was stored at
    //     tile[4*sx+k][(4*sx+k) ^ sy].
    int row_out = blockIdx.x * TILE_SZ + sy;
    int col_out = blockIdx.y * TILE_SZ + 4 * sx;

    if (row_out < w && col_out < h) {
        float4 tmp;
        tmp.x = tile[4 * sx + 0][(4 * sx + 0) ^ sy];
        tmp.y = tile[4 * sx + 1][(4 * sx + 1) ^ sy];
        tmp.z = tile[4 * sx + 2][(4 * sx + 2) ^ sy];
        tmp.w = tile[4 * sx + 3][(4 * sx + 3) ^ sy];
        ST_F4(B[row_out * h + col_out]) = tmp;
    }
}

// ===========================================================================
// Benchmark helper
// ===========================================================================
template <typename KernelFunc, typename... Args>
static double benchKernel(KernelFunc kernel, dim3 grid, dim3 block,
                          int nreps, Args... args)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    kernel<<<grid, block>>>(args...);
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
                   double tol, const char *label)
{
    for (int i = 0; i < n; ++i) {
        double diff = static_cast<double>(gpu[i]) - static_cast<double>(cpu[i]);
        if (diff < 0.0) diff = -diff;
        if (diff > tol) {
            std::cout << "  " << label << "  FAIL at [" << i << "]: GPU="
                      << gpu[i] << ", CPU=" << cpu[i] << std::endl;
            return false;
        }
    }
    std::cout << "  " << label << "  PASS" << std::endl;
    return true;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    // ------------------------------------------------------------------
    // 1. Device info
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int memClock_kHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClock_kHz,
                                       cudaDevAttrMemoryClockRate, 0));

    double bwTheor = memClock_kHz * 1e3
                     * static_cast<double>(prop.memoryBusWidth) / 8.0 * 2.0
                     / 1e9;

    std::cout << "========================================" << std::endl;
    std::cout << "  Matrix Transpose -- float4 Vectorized" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU            : " << prop.name << std::endl;
    std::cout << "  Memory clock   : " << memClock_kHz * 1e-3f << " MHz" << std::endl;
    std::cout << "  Memory bus     : " << prop.memoryBusWidth << "-bit" << std::endl;
    std::cout << "  Theoretical BW : " << bwTheor << " GB/s" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 2. Matrix dimensions
    // ------------------------------------------------------------------
    constexpr int H  = 4096;
    constexpr int W  = 4096;
    constexpr int N  = H * W;
    constexpr int TILE_SZ = 32;

    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    double dataPerRun = 2.0 * static_cast<double>(bytes);  // read + write

    std::cout << "  Matrix  : " << H << " x " << W << "  ("
              << (bytes >> 20) << " MB)" << std::endl;
    std::cout << "  Tile    : " << TILE_SZ << " x " << TILE_SZ << std::endl;
    std::cout << "  Data/run: " << dataPerRun / (1024.0 * 1024.0)
              << " MB (read + write)" << std::endl;
    std::cout << "  Threads/block: " << TILE_SZ << " x " << (TILE_SZ / 4)
              << " = " << TILE_SZ * TILE_SZ / 4 << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 3. Allocate + init host memory
    // ------------------------------------------------------------------
    auto *h_A   = new float[N];
    auto *h_B   = new float[N];
    auto *h_ref = new float[N];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i)
        h_A[i] = dist(rng);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_A, H, W, h_ref);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ------------------------------------------------------------------
    // 4. Device memory
    // ------------------------------------------------------------------
    float *d_A = nullptr;
    float *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 5. Launch configuration
    // ------------------------------------------------------------------
    dim3 blockF4(static_cast<unsigned int>(TILE_SZ),
                 static_cast<unsigned int>(TILE_SZ / 4));
    dim3 gridF4((W + TILE_SZ - 1) / TILE_SZ,
                (H + TILE_SZ - 1) / TILE_SZ);

    constexpr int NREPS = 20;

    // ------------------------------------------------------------------
    // 6. Benchmark + verify
    // ------------------------------------------------------------------
    std::cout << "--- Kernel Performance (" << NREPS
              << " reps average) ---" << std::endl;

    double kernelMs[3] = {};
    bool   pass[3]     = {};

    // v1: tile, bank conflict
    kernelMs[0] = benchKernel(
        transpose_float4_v1<TILE_SZ>, gridF4, blockF4, NREPS,
        d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[0] = verify(h_B, h_ref, N, 1e-5, "v1");
    std::cout << "       " << kernelMs[0] << " ms  |  BW "
              << dataPerRun / (kernelMs[0] * 1e-3) / 1e9 << " GB/s"
              << std::endl;

    // v2: tile + padding
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    kernelMs[1] = benchKernel(
        transpose_float4_v2<TILE_SZ>, gridF4, blockF4, NREPS,
        d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[1] = verify(h_B, h_ref, N, 1e-5, "v2");
    std::cout << "       " << kernelMs[1] << " ms  |  BW "
              << dataPerRun / (kernelMs[1] * 1e-3) / 1e9 << " GB/s"
              << std::endl;

    // v3: tile + XOR
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    kernelMs[2] = benchKernel(
        transpose_float4_v3<TILE_SZ>, gridF4, blockF4, NREPS,
        d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[2] = verify(h_B, h_ref, N, 1e-5, "v3");
    std::cout << "       " << kernelMs[2] << " ms  |  BW "
              << dataPerRun / (kernelMs[2] * 1e-3) / 1e9 << " GB/s"
              << std::endl;

    // ------------------------------------------------------------------
    // 7. Summary
    // ------------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << std::left
              << std::setw(12) << "Version"
              << std::right
              << std::setw(10) << "Time(ms)"
              << std::setw(12) << "BW(GB/s)"
              << std::setw(12) << "Eff.BW%"
              << std::setw(12) << "vs CPU/x"
              << std::endl;

    const char *labels[] = {"v1", "v2", "v3"};
    for (int i = 0; i < 3; ++i) {
        double bw = dataPerRun / (kernelMs[i] * 1e-3) / 1e9;
        std::cout << std::left  << std::setw(12) << labels[i]
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << kernelMs[i]
                  << std::setw(12) << std::setprecision(2) << bw
                  << std::setw(11) << std::setprecision(1)
                  << 100.0 * bw / bwTheor << " %"
                  << std::setw(11) << std::setprecision(1)
                  << cpuMs / kernelMs[i] << " x"
                  << (pass[i] ? "" : "  !!FAIL!!")
                  << std::endl;
    }

    std::cout << std::left  << std::setw(12) << "CPU"
              << std::right << std::fixed
              << std::setw(10) << std::setprecision(3) << cpuMs
              << std::setw(12) << "--"
              << std::setw(12) << "--"
              << std::setw(12) << "1.0 x"
              << std::endl;

    std::cout << std::endl;
    std::cout << "  Theoretical BW: " << bwTheor << " GB/s" << std::endl;

    // ------------------------------------------------------------------
    // 8. Cleanup
    // ------------------------------------------------------------------
    delete[] h_A;
    delete[] h_B;
    delete[] h_ref;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    bool allPassed = pass[0] && pass[1] && pass[2];
    std::cout << std::endl
              << (allPassed ? "All tests PASSED" : "Some tests FAILED")
              << std::endl;
    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
