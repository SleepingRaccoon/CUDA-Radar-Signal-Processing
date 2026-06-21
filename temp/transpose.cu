/*
nvcc -o transpose transpose.cu
transpose.exe
*/

#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// Error-check macro
// --------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "[ERROR] %s:%d  %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// ===========================================================================
// CPU reference
// ===========================================================================
void transpose_cpu(const float *A, int h, int w, float *B)
{
    for (int y = 0; y < w; y++)
        for (int x = 0; x < h; x++)
            B[y * h + x] = A[x * w + y];
}

// ===========================================================================
// v11: coalesced read, non-coalesced write (stride-h writes)
// ===========================================================================
__global__ void transpose_v11(const float *A, int h, int w, float *B)
{
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ty < h && tx < w)
        B[tx * h + ty] = A[ty * w + tx];
}

// ===========================================================================
// v12: coalesced write, non-coalesced read (stride-w reads)
// ===========================================================================
__global__ void transpose_v12(const float *A, int h, int w, float *B)
{
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ty < w && tx < h)
        B[ty * h + tx] = A[tx * w + ty];
}

// ===========================================================================
// v2:  shared-memory tile, coalesced both directions, bank conflict present
// ===========================================================================
template <int TILE_SZ>
__global__ void transpose_v2(const float *A, int h, int w, float *B)
{
    __shared__ float tile[TILE_SZ][TILE_SZ];

    int row = blockIdx.y * TILE_SZ + threadIdx.y;
    int col = blockIdx.x * TILE_SZ + threadIdx.x;

    if (row < h && col < w)
        tile[threadIdx.y][threadIdx.x] = A[row * w + col];

    __syncthreads();

    // B is W×H.  Output row (original col): bx*TILE_SZ + threadIdx.y
    //             Output col (original row): by*TILE_SZ + threadIdx.x
    int row_out = blockIdx.x * TILE_SZ + threadIdx.y;
    int col_out = blockIdx.y * TILE_SZ + threadIdx.x;

    if (row_out < w && col_out < h)
        B[row_out * h + col_out] = tile[threadIdx.x][threadIdx.y];
}

// ===========================================================================
// v3:  shared-memory tile + padding, no bank conflict
// ===========================================================================
template <int TILE_SZ>
__global__ void transpose_v3(const float *A, int h, int w, float *B)
{
    __shared__ float tile[TILE_SZ][TILE_SZ + 1];

    int row = blockIdx.y * TILE_SZ + threadIdx.y;
    int col = blockIdx.x * TILE_SZ + threadIdx.x;

    if (row < h && col < w)
        tile[threadIdx.y][threadIdx.x] = A[row * w + col];

    __syncthreads();

    int row_out = blockIdx.x * TILE_SZ + threadIdx.y;
    int col_out = blockIdx.y * TILE_SZ + threadIdx.x;

    if (row_out < w && col_out < h)
        B[row_out * h + col_out] = tile[threadIdx.x][threadIdx.y];
}

// ===========================================================================
// v4:  shared-memory tile + XOR indexing, no bank conflict
// ===========================================================================
template <int TILE_SZ>
__global__ void transpose_v4(const float *A, int h, int w, float *B)
{
    __shared__ float tile[TILE_SZ][TILE_SZ];

    int row = blockIdx.y * TILE_SZ + threadIdx.y;
    int col = blockIdx.x * TILE_SZ + threadIdx.x;

    if (row < h && col < w)
        tile[threadIdx.y][threadIdx.y ^ threadIdx.x] = A[row * w + col];

    __syncthreads();

    int row_out = blockIdx.x * TILE_SZ + threadIdx.y;
    int col_out = blockIdx.y * TILE_SZ + threadIdx.x;

    if (row_out < w && col_out < h)
        B[row_out * h + col_out] = tile[threadIdx.x][threadIdx.x ^ threadIdx.y];
}

// ===========================================================================
// Timing helper: run kernel with warm-up, average over nreps, return ms + BW
// ===========================================================================
template <typename KernelFunc, typename... Args>
static double benchKernel(KernelFunc kernel, dim3 grid, dim3 block,
                          int nreps, double totalBytes,
                          const char *label, Args... args)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    kernel<<<grid, block>>>(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    double totalMs = 0.0;
    for (int r = 0; r < nreps; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(args...);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += (double)ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double avgMs  = totalMs / nreps;
    double bwGBs  = totalBytes / (avgMs * 1e-3) / 1e9;

    printf("  %-10s  avg %7.3f ms  |  BW %7.2f GB/s\n", label, avgMs, bwGBs);
    return avgMs;
}

// ===========================================================================
// Correctness check
// ===========================================================================
static bool verify(const float *gpu, const float *cpu, int n,
                    double tol, const char *label)
{
    for (int i = 0; i < n; i++) {
        double diff = (double)gpu[i] - (double)cpu[i];
        if (diff < 0.0) diff = -diff;
        if (diff > tol) {
            printf("  %-10s  FAIL at [%d]: GPU=%.6f, CPU=%.6f\n",
                   label, i, gpu[i], cpu[i]);
            return false;
        }
    }
    printf("  %-10s  PASS\n", label);
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

    // Theoretical memory BW (GB/s) = clock(kHz)*1e3 * busWidth(bits)/8 * 2(DDR) / 1e9
    double bwTheoretical = memClock_kHz * 1e3
                           * (double)prop.memoryBusWidth / 8.0 * 2.0
                           / 1e9;

    printf("========================================\n");
    printf("  Matrix Transpose -- Performance Test\n");
    printf("========================================\n");
    printf("  GPU        : %s\n", prop.name);
    printf("  Mem clock  : %.0f MHz\n", memClock_kHz * 1e-3);
    printf("  Mem bus    : %d-bit\n", prop.memoryBusWidth);
    printf("  Theoretical BW: %.2f GB/s\n\n", bwTheoretical);

    // ------------------------------------------------------------------
    // 2. Matrix dimensions
    // ------------------------------------------------------------------
    const int H  = 4096;
    const int W  = 4096;
    const int N  = H * W;
    const int TILE_SZ = 32;
    size_t bytes = (size_t)N * sizeof(float);

    // Total data moved per transpose = read H*W + write W*H = 2*N*sizeof(float)
    double totalDataBytes = 2.0 * (double)bytes;

    printf("  Matrix    : %d x %d  (%zu MB)\n", H, W, bytes >> 20);
    printf("  Tile size : %d x %d\n", TILE_SZ, TILE_SZ);
    printf("  Data/transpose: %.1f MB (read + write)\n\n",
           totalDataBytes / (1024.0 * 1024.0));

    // ------------------------------------------------------------------
    // 3. Allocate + initialize
    // ------------------------------------------------------------------
    float *h_A  = new float[N];
    float *h_B  = new float[N];
    float *h_ref = new float[N];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++)
        h_A[i] = dist(rng);

    // CPU reference
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_A, H, W, h_ref);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(
        cpu_t1 - cpu_t0).count();

    float *d_A = nullptr, *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 4. Grid / block setup
    // ------------------------------------------------------------------
    dim3 blockV1(32, 8);          // 256 threads
    dim3 gridV1((W + 31) / 32, (H + 7) / 8);

    dim3 blockTile(TILE_SZ, TILE_SZ, 1);  // 32x32 = 1024 threads
    dim3 gridTile((W + TILE_SZ - 1) / TILE_SZ,
                  (H + TILE_SZ - 1) / TILE_SZ);

    const int NREPS = 20;

    // ------------------------------------------------------------------
    // 5. Benchmark all versions
    // ------------------------------------------------------------------
    double kernelMs[6] = {0};
    bool   pass[6]     = {false};

    printf("--- Kernel Performance (%d reps average) ---\n", NREPS);

    // v11
    kernelMs[0] = benchKernel(
        transpose_v11, gridV1, blockV1, NREPS, totalDataBytes,
        "v11", d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[0] = verify(h_B, h_ref, N, 1e-5, "v11");

    // v12
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    kernelMs[1] = benchKernel(
        transpose_v12, gridV1, blockV1, NREPS, totalDataBytes,
        "v12", d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[1] = verify(h_B, h_ref, N, 1e-5, "v12");

    // v2 (tile, bank conflict)
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    kernelMs[2] = benchKernel(
        transpose_v2<TILE_SZ>, gridTile, blockTile, NREPS, totalDataBytes,
        "v2", d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[2] = verify(h_B, h_ref, N, 1e-5, "v2");

    // v3 (tile + padding)
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    kernelMs[3] = benchKernel(
        transpose_v3<TILE_SZ>, gridTile, blockTile, NREPS, totalDataBytes,
        "v3", d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[3] = verify(h_B, h_ref, N, 1e-5, "v3");

    // v4 (tile + XOR)
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    kernelMs[4] = benchKernel(
        transpose_v4<TILE_SZ>, gridTile, blockTile, NREPS, totalDataBytes,
        "v4", d_A, H, W, d_B);
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    pass[4] = verify(h_B, h_ref, N, 1e-5, "v4");

    // ------------------------------------------------------------------
    // 6. Summary
    // ------------------------------------------------------------------
    printf("\n========================================\n");
    printf("  Summary\n");
    printf("========================================\n");
    printf("  %-10s  %10s  %10s  %10s  %10s\n",
           "Version", "Time (ms)", "BW (GB/s)", "Eff. BW %%", "vs CPU/x");

    const char *labels[] = {"v11","v12","v2","v3","v4"};
    for (int i = 0; i < 5; i++) {
        double bw = totalDataBytes / (kernelMs[i] * 1e-3) / 1e9;
        printf("  %-10s  %10.3f  %10.2f  %9.1f %%  %10.1f x %s\n",
               labels[i], kernelMs[i], bw,
               100.0 * bw / bwTheoretical,
               cpuMs / kernelMs[i],
               pass[i] ? "" : "!!FAIL!!");
    }

    printf("  %-10s  %10.3f  %10s  %9s  %10s\n",
           "CPU", cpuMs, "--", "--", "1.0 x");
    printf("\n  Theoretical BW: %.2f GB/s\n\n", bwTheoretical);

    // ------------------------------------------------------------------
    // 7. Cleanup
    // ------------------------------------------------------------------
    delete[] h_A;
    delete[] h_B;
    delete[] h_ref;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    // Check all passed
    bool allPassed = true;
    for (int i = 0; i < 5; i++) allPassed = allPassed && pass[i];
    printf(allPassed ? "All tests PASSED\n" : "Some tests FAILED\n");
    return allPassed ? 0 : 1;
}
