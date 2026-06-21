/*
nvcc -o template template.cu
template.exe
*/

/**
 * template_clean.cu
 *
 * Single-file version of NVIDIA's CUDA template sample.
 *
 * Core concept -- a minimal but complete CUDA program structure:
 *   1. Host memory allocation + initialization
 *   2. Device memory allocation
 *   3. Host-to-device data copy
 *   4. Kernel launch with block/thread configuration
 *   5. Device-to-host result copy
 *   6. CPU reference computation
 *   7. Result verification
 *
 * Kernel details:
 *   - Uses dynamic shared memory (extern __shared__)
 *   - __syncthreads() barriers between shared memory phases
 *   - Each element is multiplied by num_threads
 *
 *   Shared memory usage pattern:
 *     (1) load from global -> shared
 *     (2) __syncthreads()   <- ensure all threads have loaded
 *     (3) compute on shared
 *     (4) __syncthreads()   <- ensure all threads have computed
 *     (5) store shared -> global
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

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
// Kernel: load from global -> compute on shared -> store to global
// ---------------------------------------------------------------------------
//   Each block has its own shared memory region (sdata).
//   The kernel:
//     (1) Each thread copies one element from global to shared memory
//     (2) __syncthreads() ensures all copies are done
//     (3) Each thread multiplies its element by num_threads
//     (4) __syncthreads() ensures all computations are done
//     (5) Each thread writes its result back to global memory
//
//   Shared memory size = num_threads * sizeof(float),
//   passed as the 3rd argument in <<<grid, block, smem_bytes>>>
// ---------------------------------------------------------------------------
__global__ void testKernel(const float *g_idata, float *g_odata, int num_threads)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // Load from global to shared
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // Compute: multiply by number of threads per block
    sdata[tid] = (float)num_threads * sdata[tid];
    __syncthreads();

    // Store result back to global
    g_odata[tid] = sdata[tid];
}

// ---------------------------------------------------------------------------
// CPU reference: each element multiplied by len (same as GPU)
// ---------------------------------------------------------------------------
static void computeGold(float *reference, const float *idata, int len)
{
    float f_len = (float)len;
    for (int i = 0; i < len; ++i) {
        reference[i] = idata[i] * f_len;
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("template Starting...\n\n");

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

    // ------------------------------------------------------------------
    // 2. Parameters
    // ------------------------------------------------------------------
    const int num_threads = 32;
    int mem_size = (int)(sizeof(float) * num_threads);

    dim3 grid(1);
    dim3 block(num_threads);

    printf("Threads: %d\n", num_threads);
    printf("SMEM per block: %d bytes\n\n", mem_size);

    // ------------------------------------------------------------------
    // 3. Host memory: allocate + initialize
    // ------------------------------------------------------------------
    float *h_idata = new float[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        h_idata[i] = (float)i;
    }

    // ------------------------------------------------------------------
    // 4. Device memory: allocate + copy input
    // ------------------------------------------------------------------
    float *d_idata = nullptr;
    float *d_odata = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_idata, mem_size));
    CUDA_CHECK(cudaMalloc((void **)&d_odata, mem_size));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 5. Warm-up + timed kernel launch
    // ------------------------------------------------------------------
    testKernel<<<grid, block, mem_size>>>(d_idata, d_odata, num_threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for kernel launch / execution errors
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    testKernel<<<grid, block, mem_size>>>(d_idata, d_odata, num_threads);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    printf("Kernel time: %.4f ms\n", kernel_ms);

    // ------------------------------------------------------------------
    // 6. Copy result back to host
    // ------------------------------------------------------------------
    float *h_odata = new float[num_threads];
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // 7. CPU reference computation
    // ------------------------------------------------------------------
    float *reference = new float[num_threads];
    computeGold(reference, h_idata, num_threads);

    // ------------------------------------------------------------------
    // 8. Verify result
    // ------------------------------------------------------------------
    printf("\n--- Verification ---\n");
    bool passed = true;
    for (int i = 0; i < num_threads; ++i) {
        if (fabs(reference[i] - h_odata[i]) > 1e-5f) {
            printf("  FAIL at [%d]: CPU=%.4f, GPU=%.4f\n",
                   i, reference[i], h_odata[i]);
            passed = false;
            break;
        }
    }
    if (passed) {
        printf("  Result: PASS\n");
        printf("\n");
        printf("  Input -> GPU -> Output:\n");
        for (int i = 0; i < num_threads; ++i) {
            printf("    %3.0f  ->  %6.0f\n", h_idata[i], h_odata[i]);
        }
    }

    // ------------------------------------------------------------------
    // 9. Cleanup
    // ------------------------------------------------------------------
    delete[] h_idata;
    delete[] h_odata;
    delete[] reference;
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nDone!\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}