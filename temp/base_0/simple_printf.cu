/*

nvcc -o simple_printf simple_printf.cu
simple_printf.exe

*/

/**
 * simplePrintf_clean.cu
 *
 * Single-file version of NVIDIA's simplePrintf sample.
 *
 * Core concept:
 *   CUDA kernels (CC >= 2.0) support printf() directly from device code.
 *   The output is buffered in a device-side ring buffer and flushed to
 *   stdout when cudaDeviceSynchronize() is called.
 *
 *   This sample also demonstrates multi-dimensional grid and block
 *   configurations:
 *     - 2D grid     (2 x 2 blocks)
 *     - 3D block    (2 x 2 x 2 threads)
 *     -> total 4 * 8 = 32 threads, each printing its global index.
 *
 *   Global index calculation:
 *     block  = blockIdx.y * gridDim.x  + blockIdx.x       (flatten 2D grid)
 *     thread = threadIdx.z * (blockDim.x * blockDim.y)
 *            + threadIdx.y * blockDim.x
 *            + threadIdx.x                                 (flatten 3D block)
 */

#include <cstdio>
#include <cstdlib>

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
// Kernel: print thread/block index and a passed-in value
// ---------------------------------------------------------------------------
__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",
           blockIdx.y * gridDim.x + blockIdx.x,
           threadIdx.z * (blockDim.x * blockDim.y)
               + threadIdx.y * blockDim.x
               + threadIdx.x,
           val);
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    // ------------------------------------------------------------------
    // 1. Pick device 0 and query properties
    // ------------------------------------------------------------------
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        printf("No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Device %d: \"%s\" with Compute capability %d.%d\n",
           0, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");

    // ------------------------------------------------------------------
    // 2. Kernel launch: 2D grid, 3D block
    // ------------------------------------------------------------------
    //   Grid:  2 x 2 blocks = 4 blocks total
    //   Block: 2 x 2 x 2 threads = 8 threads per block
    //   Total: 4 * 8 = 32 threads
    //
    //   CC < 2.0: printf() is not supported in device code (will be NOP).
    // ------------------------------------------------------------------
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel<<<dimGrid, dimBlock>>>(10);
    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
