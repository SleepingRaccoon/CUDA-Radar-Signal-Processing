/*
    nvcc -o demo3 demo3.cu
    demo3.exe

    nvcc -o simple_occupancy simple_occupancy.cu
    simple_occupancy.exe
*/

/**
 * simpleOccupancy_clean.cu
 *
 * Single-file version of NVIDIA's simpleOccupancy sample.
 *
 * Core concept -- Occupancy:
 *   Occupancy = (active warps per SM) / (max warps per SM)
 *
 *   Each SM can run multiple thread blocks simultaneously, but limited by:
 *     (a) max blocks per SM        (hardware limit)
 *     (b) max threads per SM       (hardware limit)
 *     (c) max registers per SM     (per-thread register count x threads)
 *     (d) shared memory per SM     (per-block SMEM x blocks)
 *
 *   Higher occupancy means more warps available to hide memory latency.
 *   But occupancy alone is NOT performance -- sometimes lower occupancy
 *   with more registers per thread is faster (avoiding register spilling).
 *
 * This program:
 *   1. Queries device properties relevant to occupancy
 *   2. Scans occupancy across block sizes (32 to 1024)
 *   3. Compares performance: manual (bad) vs occupancy-optimized block size
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[ERROR] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: square each array element
// ---------------------------------------------------------------------------
//   Uses dynamic shared memory (extern __shared__) just to show how SMEM
//   affects occupancy calculations. In this case we don't actually use it.
// ---------------------------------------------------------------------------
__global__ void square_kernel(uint32_t *array, int arrayCount)
{
    // Declare dynamic shared memory (not actually used, but counts toward
    // the kernel's resource usage for occupancy computation).
    extern __shared__ int dynamicSmem[];
    (void)dynamicSmem;  // suppress unused warning

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// ---------------------------------------------------------------------------
// Compute theoretical occupancy for a given block size
// ---------------------------------------------------------------------------
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor tells us how many blocks
//   can run simultaneously on one SM, given the kernel's resource usage.
//
//   From that:
//     activeWarps = numBlocks * (blockSize / warpSize)
//     maxWarps    = maxThreadsPerSM / warpSize
//     occupancy   = activeWarps / maxWarps
// ---------------------------------------------------------------------------
static double compute_occupancy(const void *kernel, int blockSize, size_t dynamicSmem)
{
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int numBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, kernel, blockSize, dynamicSmem));

    int activeWarps = numBlocks * (blockSize / prop.warpSize);
    int maxWarps    = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    return (double)activeWarps / (double)maxWarps;
}

// ---------------------------------------------------------------------------
// Run the kernel with a given block size, measure time
// ---------------------------------------------------------------------------
static double run_kernel(uint32_t *dArray, int arrayCount,
                          int blockSize, size_t dynamicSmem)
{
    int gridSize = (arrayCount + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up: avoid cold-launch overhead in timing
    square_kernel<<<gridSize, blockSize, dynamicSmem>>>(dArray, arrayCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    square_kernel<<<gridSize, blockSize, dynamicSmem>>>(dArray, arrayCount);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (double)ms;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("========================================\n");
    printf("[ simpleOccupancy -- CUDA Occupancy ]\n");
    printf("========================================\n");

    // ------------------------------------------------------------------
    // 1. Device info
    // ------------------------------------------------------------------
    int deviceCount = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device       : %s\n", prop.name);
    printf("SMs          : %d\n", prop.multiProcessorCount);
    printf("Warp size    : %d\n", prop.warpSize);
    printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks/SM : %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Regs/SM      : %d\n", prop.regsPerMultiprocessor);

    int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    printf("Max warps/SM : %d\n\n", maxWarps);

    // ------------------------------------------------------------------
    // 2. Occupancy scan across block sizes
    // ------------------------------------------------------------------
    //   We'll compute theoretical occupancy for block sizes 32, 64, 128,
    //   256, 512, 1024 (all valid for most GPUs).
    // ------------------------------------------------------------------
    printf("--- Occupancy Scan (theoretical) ---\n");
    printf("BlockSize | Blocks/SM | ActiveWarps | Occupancy\n");
    printf("----------+-----------+-------------+----------\n");

    int blockSizes[] = {32, 64, 128, 192, 256, 384, 512, 768, 1024};
    int numSizes = (int)(sizeof(blockSizes) / sizeof(blockSizes[0]));

    for (int i = 0; i < numSizes; i++) {
        int bs = blockSizes[i];
        if (bs > prop.maxThreadsPerBlock) continue;

        int numBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, (const void *)square_kernel, bs, 0));

        int activeWarps = numBlocks * (bs / prop.warpSize);
        double occ = (double)activeWarps / (double)maxWarps;

        // Visual bar
        int barLen = (int)(occ * 20.0 + 0.5);
        printf("  %4d    |   %3d    |    %3d     |  %5.1f%%  ",
               bs, numBlocks, activeWarps, occ * 100.0);
        for (int b = 0; b < barLen; b++) printf("#");
        printf("\n");
    }
    printf("\n");

    // ------------------------------------------------------------------
    // 3. Performance comparison: manual vs occupancy-optimized
    // ------------------------------------------------------------------
    //   We use the same square kernel on a 4M-element array.
    //   - Manual:   blockSize = 32 (very low occupancy)
    //   - Auto:     let cudaOccupancyMaxPotentialBlockSize choose
    // ------------------------------------------------------------------
    const int arrayCount = 4 * 1024 * 1024;   // 4M uint32_t = 16 MB
    size_t    arrayBytes = arrayCount * sizeof(uint32_t);

    printf("--- Performance Comparison ---\n");
    printf("Array size: %d elements (%zu MB)\n\n", arrayCount, arrayBytes >> 20);

    // Prepare host data
    uint32_t *hArray = new uint32_t[arrayCount];
    for (int i = 0; i < arrayCount; i++) {
        hArray[i] = (uint32_t)i;
    }

    // Device memory
    uint32_t *dArray = nullptr;
    CUDA_CHECK(cudaMalloc(&dArray, arrayBytes));
    CUDA_CHECK(cudaMemcpy(dArray, hArray, arrayBytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 3a. Manual configuration: blockSize = 32
    // ------------------------------------------------------------------
    printf("[ Manual configuration ]\n");
    printf("  Block size: 32 (fixed)\n");

    int manualBlockSize = 32;
    double manualOcc = compute_occupancy(
        (const void *)square_kernel, manualBlockSize, 0);
    double manualTime = run_kernel(dArray, arrayCount, manualBlockSize, 0);
    printf("  Occupancy : %.1f%%\n", manualOcc * 100.0);
    printf("  Time      : %.3f ms\n\n", manualTime);

    // Reload original data before auto test
    for (int i = 0; i < arrayCount; i++) hArray[i] = (uint32_t)i;
    CUDA_CHECK(cudaMemcpy(dArray, hArray, arrayBytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 3b. Automatic occupancy-based configuration
    // ------------------------------------------------------------------
    //   cudaOccupancyMaxPotentialBlockSize suggests the block size that
    //   achieves the best theoretical occupancy, and the minimum grid
    //   size needed to fill the GPU.
    // ------------------------------------------------------------------
    printf("[ Automatic (occupancy-based) configuration ]\n");

    int autoBlockSize = 0;
    int autoMinGrid   = 0;
    size_t dynamicSmem = 0;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &autoMinGrid, &autoBlockSize,
        square_kernel, dynamicSmem, arrayCount));

    double autoOcc = compute_occupancy(
        (const void *)square_kernel, autoBlockSize, 0);
    double autoTime = run_kernel(dArray, arrayCount, autoBlockSize, dynamicSmem);

    printf("  Suggested block size : %d\n", autoBlockSize);
    printf("  Min grid size (full GPU) : %d\n", autoMinGrid);
    printf("  Occupancy : %.1f%%\n", autoOcc * 100.0);
    printf("  Time      : %.3f ms\n\n", autoTime);

    // ------------------------------------------------------------------
    // 4. Comparison summary
    // ------------------------------------------------------------------
    printf("========================================\n");
    printf("Comparison Summary\n");
    printf("========================================\n");
    printf("  Manual (block=%4d):  %5.1f%% occ,  %.3f ms\n",
           manualBlockSize, manualOcc * 100.0, manualTime);
    printf("  Auto   (block=%4d):  %5.1f%% occ,  %.3f ms\n",
           autoBlockSize, autoOcc * 100.0, autoTime);
    printf("\n");
    printf("  Speedup from better occupancy: %.2fx\n",
           manualTime / autoTime);
    printf("\n");

    // ------------------------------------------------------------------
    // 5. Verification
    // ------------------------------------------------------------------
    printf("--- Verification ---\n");

    // Reset and re-run with auto config for verification
    for (int i = 0; i < arrayCount; i++) hArray[i] = (uint32_t)i;
    CUDA_CHECK(cudaMemcpy(dArray, hArray, arrayBytes, cudaMemcpyHostToDevice));

    int verifyGrid = (arrayCount + autoBlockSize - 1) / autoBlockSize;
    square_kernel<<<verifyGrid, autoBlockSize, 0>>>(dArray, arrayCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hArray, dArray, arrayBytes, cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < arrayCount; i++) {
        uint32_t expected = (uint32_t)i * (uint32_t)i;
        if (hArray[i] != expected) {
            printf("  FAIL at [%d]: got %u, expected %u\n",
                   i, hArray[i], expected);
            pass = false;
            break;
        }
    }
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    // ------------------------------------------------------------------
    // 6. Cleanup
    // ------------------------------------------------------------------
    delete[] hArray;
    CUDA_CHECK(cudaFree(dArray));

    printf("\nDone!\n");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
