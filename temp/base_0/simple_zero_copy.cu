/*
nvcc -o simple_zero_copy simple_zero_copy.cu
simple_zero_copy.exe

cd temp
nvcc -o demo2 demo2.cu
demo2.exe
*/

/**
 * Single-file version of NVIDIA's simpleZeroCopy sample.
 *
 * Core concept -- Zero-Copy (Mapped Memory):
 *   Normally GPU works like this:
 *     CPU allocates memory -> cudaMalloc on device -> cudaMemcpy H2D ->
 *     kernel runs on device memory -> cudaMemcpy D2H
 *
 *   Zero-Copy eliminates the explicit memcpy by mapping host memory
 *   directly into the GPU's address space. The kernel reads/writes
 *   host memory over the PCIe bus on every access -- no memcpy needed.
 *
 *   Analog: Think of it like mmap() on the CPU side. The GPU sees
 *   host pages as if they were device memory.
 *
 *   TWO strategies for mapped memory:
 *     A) cudaHostAlloc + cudaHostAllocMapped
 *        - CUDA allocates pinned + mapped memory in one call
 *        - cudaHostGetDevicePointer returns the GPU-side pointer
 *
 *     B) malloc + cudaHostRegister + cudaHostRegisterMapped
 *        - Allocate generic page-aligned memory, then pin+map it
 *        - Useful when memory is already allocated by other code
 *        - Requires 4K-aligned memory (page boundary)
 *
 *   TRADEOFF: When to use Zero-Copy?
 *     PRO: No explicit memcpy, simpler code, good for small/irregular data
 *     CON: Every GPU access goes over PCIe -> higher latency, lower bandwidth
 *          than local device memory
 *
 *   Rule of thumb:
 *     - Small data, accessed once: zero-copy is fine
 *     - Large data, accessed repeatedly: use explicit memcpy to device memory
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

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

#define MEMORY_ALIGNMENT 4096

#define ALIGN_UP(ptr, alignment)                                            \
    ((void *)((((size_t)(ptr)) + ((alignment) - 1)) & ~((alignment) - 1)))

// ===========================================================================
// Kernel: vector add -- reads/writes mapped host memory directly
// ===========================================================================
//   a, b, c are DEVICE-side pointers obtained from cudaHostGetDevicePointer.
//   They point to host memory that has been mapped into GPU address space.
//   Every load/store goes across PCIe -- no explicit memcpy needed.
// ===========================================================================
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[simpleZeroCopy] - Starting...\n");

    // ------------------------------------------------------------------
    // 1. Pick device 0 and verify mapped memory support
    // ------------------------------------------------------------------
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        printf("No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device: \"%s\", SM %d.%d\n", prop.name, prop.major, prop.minor);

    // Mapped memory requires hardware support (CC >= 1.2, and the
    // device must set canMapHostMemory = 1).
    if (!prop.canMapHostMemory) {
        printf("ERROR: Device does not support mapping CPU host memory!\n");
        return EXIT_SUCCESS;
    }

    // cudaDeviceMapHost: enables the GPU to access mapped host memory.
    // Must be set BEFORE any mapped allocation.
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // ------------------------------------------------------------------
    // 2. Choose allocation strategy
    // ------------------------------------------------------------------
    //   Mac does not support generic pinning via cudaHostRegister.
    //   On Windows/Linux, use Strategy B (generic) to demonstrate both.
    // ------------------------------------------------------------------
#if defined(__APPLE__) || defined(MACOSX)
    bool useGenericPinning = false;
#else
    bool useGenericPinning = true;
#endif

    if (useGenericPinning) {
        printf("Using generic system memory (malloc + cudaHostRegisterMapped)\n\n");
    } else {
        printf("Using cudaHostAllocMapped\n\n");
    }

    // ------------------------------------------------------------------
    // 3. Allocate mapped host memory
    // ------------------------------------------------------------------
    int nelem = 1024 * 1024;          // 1M floats
    size_t bytes = nelem * sizeof(float);

    float *a = nullptr, *b = nullptr, *c = nullptr;
    float *a_base = nullptr, *b_base = nullptr, *c_base = nullptr;

    if (useGenericPinning) {
        // ---- Strategy B: malloc + align + cudaHostRegisterMapped ----
        //
        // cudaHostRegister requires 4K-aligned memory. We overallocate
        // by MEMORY_ALIGNMENT bytes, then shift the pointer to the next
        // 4K boundary. The original (unaligned) pointer is saved for
        // later free().
        //
        a_base = (float *)malloc(bytes + MEMORY_ALIGNMENT);
        b_base = (float *)malloc(bytes + MEMORY_ALIGNMENT);
        c_base = (float *)malloc(bytes + MEMORY_ALIGNMENT);

        a = (float *)ALIGN_UP(a_base, MEMORY_ALIGNMENT);
        b = (float *)ALIGN_UP(b_base, MEMORY_ALIGNMENT);
        c = (float *)ALIGN_UP(c_base, MEMORY_ALIGNMENT);

        printf("  malloc:       %p -> aligned to %p (%zu MB)\n",
               (void *)a_base, (void *)a, bytes >> 20);

        // Pin + map: make this host memory visible to the GPU
        CUDA_CHECK(cudaHostRegister(a, bytes, cudaHostRegisterMapped));
        CUDA_CHECK(cudaHostRegister(b, bytes, cudaHostRegisterMapped));
        CUDA_CHECK(cudaHostRegister(c, bytes, cudaHostRegisterMapped));

        printf("  cudaHostRegisterMapped: GPU can now access these buffers\n");
    } else {
        // ---- Strategy A: cudaHostAlloc with cudaHostAllocMapped ----
        //
        // Single call: allocate pinned memory AND map it for GPU access.
        //
        unsigned int flags = cudaHostAllocMapped;

        CUDA_CHECK(cudaHostAlloc((void **)&a, bytes, flags));
        CUDA_CHECK(cudaHostAlloc((void **)&b, bytes, flags));
        CUDA_CHECK(cudaHostAlloc((void **)&c, bytes, flags));

        printf("  cudaHostAllocMapped: %p (%zu MB)\n", (void *)a, bytes >> 20);
    }

    // ------------------------------------------------------------------
    // 4. Initialize input vectors
    // ------------------------------------------------------------------
    for (int i = 0; i < nelem; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    // ------------------------------------------------------------------
    // 5. Get GPU-side pointers for the mapped host memory
    // ------------------------------------------------------------------
    //   cudaHostGetDevicePointer translates a host pointer (a) into a
    //   device pointer (d_a) that the GPU can dereference. Both pointers
    //   refer to the SAME physical memory -- no copy happens.
    // ------------------------------------------------------------------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));

    printf("\n  Host  pointer a = %p,  Device pointer d_a = %p\n",
           (void *)a, (void *)d_a);
    printf("  (Both refer to the same physical memory page)\n");

    // ------------------------------------------------------------------
    // 6. Launch kernel (reads/writes mapped host memory via PCIe)
    // ------------------------------------------------------------------
    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));

    printf("\n> Launching vectorAddGPU (reads/writes host memory directly)...\n");

    // Warm-up
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    printf("  Kernel time: %.3f ms\n", kernel_ms);

    // ------------------------------------------------------------------
    // 7. Verify: CPU reads the SAME memory the GPU just wrote
    // ------------------------------------------------------------------
    //   No cudaMemcpy needed! The GPU wrote directly into c[] (host memory).
    //   We just read c[] on the CPU side.
    // ------------------------------------------------------------------
    printf("\n> Verifying results (no cudaMemcpy needed!)...\n");

    float errorNorm = 0.0f, refNorm = 0.0f;

    for (int i = 0; i < nelem; i++) {
        float ref  = a[i] + b[i];
        float diff = c[i] - ref;
        errorNorm += diff * diff;
        refNorm   += ref * ref;
    }

    errorNorm = sqrtf(errorNorm);
    refNorm   = sqrtf(refNorm);

    printf("  L2 error: %e\n", errorNorm);
    printf("  L2 norm:  %e\n", refNorm);
    printf("  Rel error: %e\n", errorNorm / refNorm);

    bool pass = (errorNorm / refNorm < 1.0e-6f);

    // ------------------------------------------------------------------
    // 8. Cleanup
    // ------------------------------------------------------------------
    printf("\n> Releasing memory...\n");

    if (useGenericPinning) {
        CUDA_CHECK(cudaHostUnregister(a));
        CUDA_CHECK(cudaHostUnregister(b));
        CUDA_CHECK(cudaHostUnregister(c));
        free(a_base);
        free(b_base);
        free(c_base);
    } else {
        CUDA_CHECK(cudaFreeHost(a));
        CUDA_CHECK(cudaFreeHost(b));
        CUDA_CHECK(cudaFreeHost(c));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nDone!  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}