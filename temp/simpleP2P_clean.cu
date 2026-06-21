/**
 * simpleP2P_clean.cu
 *
 * Single-file version of NVIDIA's simpleP2P sample.
 *
 * Core concept -- Peer-to-Peer (P2P) GPU memory access:
 *   GPUs connected via NVLink or PCIe can directly read/write each other's
 *   device memory without CPU involvement. Combined with UVA (Unified
 *   Virtual Addressing), a pointer from any GPU is valid on any GPU --
 *   the driver automatically routes memory accesses across the interconnect.
 *
 *   This enables three powerful patterns:
 *     1. cudaMemcpy(gpu1_ptr, gpu0_ptr, size, cudaMemcpyDefault)
 *        GPU-to-GPU copy without staging through CPU memory.
 *
 *     2. Kernel on GPU1 reads from GPU0's memory directly:
 *        cudaSetDevice(1); kernel<<<...>>>(gpu0_ptr, gpu1_ptr);
 *        The kernel dereferences gpu0_ptr and the hardware fetches
 *        data over NVLink/PCIe transparently.
 *
 *     3. Multi-GPU ping-pong pipelines without any CPU memcpy.
 *
 *   Prerequisites:
 *     - 64-bit OS + application (required for UVA)
 *     - Multiple GPUs with P2P capability
 *     - cudaDeviceEnablePeerAccess() to establish the connection
 *
 *   This sample:
 *     1. Queries P2P capability across all GPU pairs
 *     2. Enables P2P on the first capable pair
 *     3. Benchmarks P2P memcpy bandwidth (ping-pong, 100 iterations)
 *     4. Runs kernel on GPU1 reading GPU0's data (P2P kernel access)
 *     5. Runs kernel on GPU0 reading GPU1's data (reverse direction)
 *     6. Verifies results
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

// ===========================================================================
// Kernel: multiply each element by 2
// ===========================================================================
//   src may be on a DIFFERENT GPU than the one executing this kernel.
//   With P2P enabled, the hardware transparently reads src over NVLink/PCIe.
//   dst is always on the local GPU (explicitly allocated there).
// ===========================================================================
__global__ void SimpleKernel(const float *src, float *dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}

// ===========================================================================
// Check 64-bit build -- UVA requires 64-bit addressing
// ===========================================================================
static inline bool isAppBuiltAs64()
{
    return sizeof(void *) == 8;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[simpleP2P] - Starting...\n");

    // ------------------------------------------------------------------
    // 1. 64-bit check: P2P with UVA requires 64-bit
    // ------------------------------------------------------------------
    if (!isAppBuiltAs64()) {
        printf("simpleP2P requires 64-bit OS and application. "
               "Test waived.\n");
        return EXIT_SUCCESS;
    }

    // ------------------------------------------------------------------
    // 2. Check for multiple GPUs
    // ------------------------------------------------------------------
    printf("Checking for multiple GPUs...\n");
    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    printf("CUDA-capable device count: %d\n", gpuCount);

    if (gpuCount < 2) {
        printf("Two or more GPUs with P2P capability are required. "
               "Test waived.\n");
        return EXIT_SUCCESS;
    }

    // ------------------------------------------------------------------
    // 3. Query device properties and P2P capability matrix
    // ------------------------------------------------------------------
    cudaDeviceProp prop[64];  // max 64 GPUs
    for (int i = 0; i < gpuCount; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop[i], i));
    }

    printf("\nChecking GPU(s) for peer-to-peer memory access...\n");

    int p2pGPU[2] = {-1, -1};  // first capable pair found
    int canAccess = 0;

    // Scan all GPU pairs for P2P capability
    for (int i = 0; i < gpuCount; i++) {
        for (int j = 0; j < gpuCount; j++) {
            if (i == j) continue;

            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, i, j));
            printf("> Peer access %s (GPU%d) -> %s (GPU%d): %s\n",
                   prop[i].name, i, prop[j].name, j,
                   canAccess ? "Yes" : "No");

            if (canAccess && p2pGPU[0] == -1) {
                p2pGPU[0] = i;
                p2pGPU[1] = j;
            }
        }
    }

    if (p2pGPU[0] == -1) {
        printf("No P2P-capable GPU pair found. Test waived.\n");
        return EXIT_SUCCESS;
    }

    int gpu0 = p2pGPU[0];
    int gpu1 = p2pGPU[1];
    printf("\nUsing GPU pair: GPU%d <-> GPU%d\n", gpu0, gpu1);

    // ------------------------------------------------------------------
    // 4. Enable peer access (bidirectional)
    // ------------------------------------------------------------------
    //   cudaDeviceEnablePeerAccess must be called on BOTH GPUs,
    //   each enabling access to the other. This establishes the
    //   P2P mapping in the GPU page tables.
    // ------------------------------------------------------------------
    printf("Enabling peer access between GPU%d and GPU%d...\n", gpu0, gpu1);

    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu1, 0));

    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu0, 0));

    // ------------------------------------------------------------------
    // 5. Allocate buffers on both GPUs and host
    // ------------------------------------------------------------------
    size_t bufSize = 1024 * 1024 * 16 * sizeof(float);  // 64 MB
    printf("Allocating buffers (%d MB on GPU%d, GPU%d, and CPU)...\n",
           (int)(bufSize / (1024 * 1024)), gpu0, gpu1);

    CUDA_CHECK(cudaSetDevice(gpu0));
    float *g0 = nullptr;
    CUDA_CHECK(cudaMalloc(&g0, bufSize));

    CUDA_CHECK(cudaSetDevice(gpu1));
    float *g1 = nullptr;
    CUDA_CHECK(cudaMalloc(&g1, bufSize));

    // Host buffer: cudaMallocHost (pinned), automatically portable with UVA
    float *h0 = nullptr;
    CUDA_CHECK(cudaMallocHost(&h0, bufSize));

    // ------------------------------------------------------------------
    // 6. P2P memcpy bandwidth benchmark (ping-pong)
    // ------------------------------------------------------------------
    //   With UVA enabled, cudaMemcpyDefault automatically determines
    //   the direction from the pointer values -- no need to specify
    //   cudaMemcpyDeviceToDevice or cudaMemcpyPeer.
    // ------------------------------------------------------------------
    printf("Creating event handles...\n");
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(g1, g0, bufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(g0, g1, bufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed: 100 ping-pong copies
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    for (int i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            CUDA_CHECK(cudaMemcpy(g1, g0, bufSize, cudaMemcpyDefault));
        } else {
            CUDA_CHECK(cudaMemcpy(g0, g1, bufSize, cudaMemcpyDefault));
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float timeMemcpy = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&timeMemcpy, startEvent, stopEvent));

    // Bandwidth: 100 copies * 64 MB / time
    // Convert: GB = 1024^3 bytes
    double totalGB = (100.0 * (double)bufSize) / (1024.0 * 1024.0 * 1024.0);
    double bw = totalGB / (timeMemcpy / 1000.0);

    printf("P2P memcpy GPU%d <-> GPU%d: %.2f ms,  %.2f GB/s\n",
           gpu0, gpu1, timeMemcpy, bw);

    // ------------------------------------------------------------------
    // 7. Prepare host buffer + copy to GPU0
    // ------------------------------------------------------------------
    printf("Preparing host buffer and memcpy to GPU%d...\n", gpu0);

    int elemCount = (int)(bufSize / sizeof(float));
    for (int i = 0; i < elemCount; i++) {
        h0[i] = (float)(i % 4096);
    }

    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaMemcpy(g0, h0, bufSize, cudaMemcpyDefault));

    // ------------------------------------------------------------------
    // 8. Kernel launch configuration
    // ------------------------------------------------------------------
    dim3 threads(512, 1);
    dim3 blocks(elemCount / threads.x, 1);

    // ------------------------------------------------------------------
    // 9. Run kernel on GPU1, reading from GPU0, writing to GPU1
    // ------------------------------------------------------------------
    //   This is the key P2P feature: a kernel on GPU1 dereferences
    //   a pointer to GPU0's memory. The hardware fetches data over
    //   NVLink/PCIe without any explicit cudaMemcpy.
    //
    //   g0 was allocated on GPU0, but here we pass it to a kernel
    //   running on GPU1. UVA makes this transparent.
    // ------------------------------------------------------------------
    printf("Run kernel on GPU%d, reading GPU%d's data, writing to GPU%d...\n",
           gpu1, gpu0, gpu1);

    CUDA_CHECK(cudaSetDevice(gpu1));
    SimpleKernel<<<blocks, threads>>>(g0, g1);   // g0 is on GPU0!
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 10. Run kernel on GPU0, reading from GPU1, writing to GPU0
    // ------------------------------------------------------------------
    //   Reverse direction: GPU0 reads GPU1's memory.
    //   g1 was allocated on GPU1, but now used from GPU0.
    // ------------------------------------------------------------------
    printf("Run kernel on GPU%d, reading GPU%d's data, writing to GPU%d...\n",
           gpu0, gpu1, gpu0);

    CUDA_CHECK(cudaSetDevice(gpu0));
    SimpleKernel<<<blocks, threads>>>(g1, g0);   // g1 is on GPU1!
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 11. Copy result back to host and verify
    // ------------------------------------------------------------------
    //   Final value: each element went through TWO kernels: *2 then *2 = *4.
    //   Initial = i % 4096, expected = (i % 4096) * 4.
    // ------------------------------------------------------------------
    printf("Copy data back to host from GPU%d and verify...\n", gpu0);
    CUDA_CHECK(cudaMemcpy(h0, g0, bufSize, cudaMemcpyDefault));

    int errorCount = 0;
    for (int i = 0; i < elemCount; i++) {
        float expected = (float)(i % 4096) * 2.0f * 2.0f;
        if (h0[i] != expected) {
            printf("  FAIL @ [%d]: got %f, expected %f\n",
                   i, h0[i], expected);
            if (++errorCount > 10) {
                printf("  (too many errors, stopping verification)\n");
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // 12. Disable peer access
    // ------------------------------------------------------------------
    printf("Disabling peer access...\n");
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(gpu1));
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(gpu0));

    // ------------------------------------------------------------------
    // 13. Cleanup
    // ------------------------------------------------------------------
    printf("Shutting down...\n");
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaFree(g0));
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaFree(g1));
    CUDA_CHECK(cudaFreeHost(h0));

    if (errorCount != 0) {
        printf("Test FAILED!\n");
        return EXIT_FAILURE;
    }

    printf("Test PASSED\n");
    return EXIT_SUCCESS;
}
