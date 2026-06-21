/*
nvcc -o simple_assert simple_assert.cu
simple_assert.exe
*/

/**
 * simpleAssert_clean.cu
 *
 * Single-file version of NVIDIA's simpleAssert sample.
 *
 * Core concept:
 *   CUDA kernels (CC >= 2.0) support the standard C assert() macro in
 *   device code. When an assertion fails:
 *     (a) the GPU thread that hit it stops executing
 *     (b) the error is reported asynchronously
 *     (c) cudaDeviceSynchronize() returns cudaErrorAssert
 *     (d) the kernel output buffer contains the assertion info printed
 *         to stdout (file, line, thread/block index, condition)
 *
 *   This is extremely useful for debugging kernels -- catch out-of-range
 *   indices, invalid assumptions, etc. without adding manual printf guards.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <cuda_runtime.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

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
// Kernel: assert that every thread's global index is less than N
// ---------------------------------------------------------------------------
//   With Nblocks=2, Nthreads=32, N=60: 64 threads total, 4 will fail assert.
//   Threads 60-63 hit gtid >= N and trigger the assertion.
// ---------------------------------------------------------------------------
__global__ void testKernel(int N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gtid < N);
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("simpleAssert starting...\n");

    // ------------------------------------------------------------------
    // 1. OS detection
    // ------------------------------------------------------------------
#ifndef _WIN32
    utsname OS_System_Type;
    uname(&OS_System_Type);

    printf("OS_System_Type.release = %s\n", OS_System_Type.release);

    if (!strcasecmp(OS_System_Type.sysname, "Darwin")) {
        printf("simpleAssert is not currently supported on Mac OSX\n\n");
        return EXIT_SUCCESS;
    } else {
        printf("OS Info: <%s>\n\n", OS_System_Type.version);
    }
#endif

    // ------------------------------------------------------------------
    // 2. Pick device 0 and query properties
    // ------------------------------------------------------------------
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        printf("No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));

    // ------------------------------------------------------------------
    // 3. Kernel launch: 2 blocks x 32 threads = 64 threads
    // ------------------------------------------------------------------
    //   N = 60: threads 0-59 pass, threads 60-63 fail the assert.
    //   The assertion failures are printed to stdout when we sync.
    // ------------------------------------------------------------------
    dim3 dimGrid(2);
    dim3 dimBlock(32);

    printf("Launch kernel to generate assertion failures\n");
    testKernel<<<dimGrid, dimBlock>>>(60);

    // Synchronize: flushes assert output AND returns cudaErrorAssert
    printf("\n-- Begin assert output\n\n");
    cudaError_t error = cudaDeviceSynchronize();
    printf("\n-- End assert output\n\n");

    // Check for failed asserts
    if (error == cudaErrorAssert) {
        printf("Device assert failed as expected, "
               "CUDA error message is: %s\n\n",
               cudaGetErrorString(error));
    }

    bool passed = (error == cudaErrorAssert);

    printf("simpleAssert completed, returned %s\n",
           passed ? "OK" : "ERROR!");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
