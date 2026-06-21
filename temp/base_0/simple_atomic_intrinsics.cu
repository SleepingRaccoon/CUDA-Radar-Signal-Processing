/*
nvcc -o simple_atomic_intrinsics simple_atomic_intrinsics.cu
simple_atomic_intrinsics.exe
*/

/**
 * simple_atomic_intrinsics.cu
 *
 * Single-file version of NVIDIA's simpleAtomicIntrinsics sample.
 *
 * Core concept -- Atomic operations:
 *   When multiple threads read-modify-write the SAME memory location
 *   simultaneously, a classic data race occurs: thread A reads X, thread B
 *   reads X, both write X+1 -> result is X+1 instead of X+2.
 *
 *   Atomic operations solve this by making the read-modify-write cycle
 *   indivisible at the hardware level. No thread can interleave.
 *
 *   This kernel demonstrates ALL 11 atomic intrinsics on a shared 11-element
 *   array. 64 blocks x 256 threads = 16384 threads all hammer the same
 *   11 addresses simultaneously. The CPU simulates the same operations
 *   serially to verify correctness.
 *
 *   Atomic operations tested:
 *     [0]  atomicAdd  -- addition
 *     [1]  atomicSub  -- subtraction
 *     [2]  atomicExch -- exchange (unconditional replace)
 *     [3]  atomicMax  -- maximum
 *     [4]  atomicMin  -- minimum
 *     [5]  atomicInc  -- wrap-around increment (modulo limit+1)
 *     [6]  atomicDec  -- wrap-around decrement
 *     [7]  atomicCAS  -- compare-and-swap (conditional replace)
 *     [8]  atomicAnd  -- bitwise AND
 *     [9]  atomicOr   -- bitwise OR
 *     [10] atomicXor  -- bitwise XOR
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
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

// ===========================================================================
// Kernel: each thread performs 11 atomic operations on 11 shared addresses
// ===========================================================================
//   g_odata is an 11-element array in global memory.
//   Every thread (16384 of them) does atomic ops on the SAME 11 addresses.
//   The atomic hardware serializes concurrent access correctly.
// ===========================================================================
__global__ void testKernel(int *g_odata)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // ---- Arithmetic atomics ----

    // [0] atomicAdd: every thread adds 10
    //     Final = 16384 * 10 = 163840
    atomicAdd(&g_odata[0], 10);

    // [1] atomicSub: every thread subtracts 10
    //     Final = -163840
    atomicSub(&g_odata[1], 10);

    // [2] atomicExch: unconditionally replace with tid
    //     Final = some thread's tid (last writer wins, non-deterministic)
    atomicExch(&g_odata[2], tid);

    // [3] atomicMax: store the maximum tid seen
    //     Final = 16383
    atomicMax(&g_odata[3], tid);

    // [4] atomicMin: store the minimum tid seen
    //     Final = 0
    atomicMin(&g_odata[4], tid);

    // [5] atomicInc: (old >= 17) ? 0 : old+1
    //     Wraps every 18 increments (0..17). 16384 mod 18 = 4.
    //     Final = 4
    atomicInc((unsigned int *)&g_odata[5], 17);

    // [6] atomicDec: (old==0 || old>137) ? 137 : old-1
    //     Wraps every 138 decrements. 16384 mod 138 = 100 steps from 137.
    //     Final = 38
    atomicDec((unsigned int *)&g_odata[6], 137);

    // [7] atomicCAS: if g_odata[7]==tid-1, set to tid
    //     Forms a relay chain: 0->1->2->... but thread ordering is random.
    //     Final = some tid in [0, 16384) (chain breaks at race boundary)
    atomicCAS(&g_odata[7], tid - 1, tid);

    // ---- Bitwise atomics ----

    // [8] atomicAnd: bitwise AND with (2*tid+7)
    //     Final = 0xff & (2*0+7) & (2*1+7) & ...  (order irrelevant for AND)
    atomicAnd(&g_odata[8], 2 * tid + 7);

    // [9] atomicOr: bitwise OR with (1 << tid)
    //     Each thread sets one bit. All 32 bits get set.
    //     Final = 0xFFFFFFFF = -1 (signed)
    atomicOr(&g_odata[9], 1 << tid);

    // [10] atomicXor: bitwise XOR with tid
    //      XOR(0..16383) = 0 (property of XOR over consecutive ints).
    //      Initial 0xff XOR 0 = 0xff.
    //      Final = 0xff
    atomicXor(&g_odata[10], tid);
}

// ===========================================================================
// CPU reference: simulate the same atomic operations serially
// ===========================================================================
//   Since the CPU runs one thread at a time, the interleaving is fixed
//   (deterministic). The GPU interleaving is non-deterministic, but for
//   commutative operations (Add, And, Or, Xor, Max, Min) the result must
//   match the CPU. For non-commutative ones (Exch, CAS, Inc, Dec), we only
//   verify the result is in a valid range.
// ===========================================================================
static bool computeGold(const int *gpuData, int len)
{
    // ---- [0] atomicAdd ----
    int val = 0;
    for (int i = 0; i < len; ++i) val += 10;
    if (val != gpuData[0]) {
        printf("  atomicAdd failed: CPU=%d, GPU=%d\n", val, gpuData[0]);
        return false;
    }

    // ---- [1] atomicSub ----
    val = 0;
    for (int i = 0; i < len; ++i) val -= 10;
    if (val != gpuData[1]) {
        printf("  atomicSub failed: CPU=%d, GPU=%d\n", val, gpuData[1]);
        return false;
    }

    // ---- [2] atomicExch: result should be in [0, len) ----
    if (gpuData[2] < 0 || gpuData[2] >= len) {
        printf("  atomicExch failed: GPU=%d, expected in [0, %d)\n",
               gpuData[2], len);
        return false;
    }

    // ---- [3] atomicMax: result should be len-1 ----
    int maxVal = 0;
    for (int i = 0; i < len; ++i)
        maxVal = (i > maxVal) ? i : maxVal;
    if (maxVal != gpuData[3]) {
        printf("  atomicMax failed: CPU=%d, GPU=%d\n", maxVal, gpuData[3]);
        return false;
    }

    // ---- [4] atomicMin: result should be 0 ----
    int minVal = len;
    for (int i = 0; i < len; ++i)
        minVal = (i < minVal) ? i : minVal;
    if (minVal != gpuData[4]) {
        printf("  atomicMin failed: CPU=%d, GPU=%d\n", minVal, gpuData[4]);
        return false;
    }

    // ---- [5] atomicInc: wrap-around, 16384 mod 18 = 4 ----
    int limit = 17;
    val = 0;
    for (int i = 0; i < len; ++i)
        val = (val >= limit) ? 0 : val + 1;
    if (val != gpuData[5]) {
        printf("  atomicInc failed: CPU=%d, GPU=%d\n", val, gpuData[5]);
        return false;
    }

    // ---- [6] atomicDec: wrap-around ----
    limit = 137;
    val   = 0;
    for (int i = 0; i < len; ++i)
        val = ((val == 0) || (val > limit)) ? limit : val - 1;
    if (val != gpuData[6]) {
        printf("  atomicDec failed: CPU=%d, GPU=%d\n", val, gpuData[6]);
        return false;
    }

    // ---- [7] atomicCAS: relay chain, final in [0, len) ----
    if (gpuData[7] < 0 || gpuData[7] >= len) {
        printf("  atomicCAS failed: GPU=%d, expected in [0, %d)\n",
               gpuData[7], len);
        return false;
    }

    // ---- [8] atomicAnd: deterministic ----
    val = 0xff;
    for (int i = 0; i < len; ++i)
        val &= (2 * i + 7);
    if (val != gpuData[8]) {
        printf("  atomicAnd failed: CPU=%d, GPU=%d\n", val, gpuData[8]);
        return false;
    }

    // ---- [9] atomicOr: all 32 bits set -> -1 signed ----
    val = 0;
    for (int i = 0; i < len; ++i)
        val |= (1 << i);
    if (val != gpuData[9]) {
        printf("  atomicOr failed: CPU=%d, GPU=%d\n", val, gpuData[9]);
        return false;
    }

    // ---- [10] atomicXor: deterministic ----
    val = 0xff;
    for (int i = 0; i < len; ++i)
        val ^= i;
    if (val != gpuData[10]) {
        printf("  atomicXor failed: CPU=%d, GPU=%d\n", val, gpuData[10]);
        return false;
    }

    return true;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[simpleAtomicIntrinsics] - Starting...\n");

    // ------------------------------------------------------------------
    // 1. Pick device 0
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
    printf("Device: \"%s\", SM %d.%d\n\n",
           props.name, props.major, props.minor);

    // ------------------------------------------------------------------
    // 2. Parameters
    // ------------------------------------------------------------------
    int numThreads = 256;
    int numBlocks  = 64;
    int numData    = 11;            // one element per atomic operation
    int memSize    = (int)(sizeof(int) * numData);
    int totalThreads = numThreads * numBlocks;

    printf("Threads: %d blocks x %d = %d threads\n",
           numBlocks, numThreads, totalThreads);
    printf("Data: %d elements (%d bytes)\n\n", numData, memSize);

    // ------------------------------------------------------------------
    // 3. Allocate host memory (pinned for async memcpy)
    // ------------------------------------------------------------------
    int *hOData = nullptr;
    CUDA_CHECK(cudaMallocHost((void **)&hOData, memSize));

    // Initialize: all zeros, except [8] and [10] = 0xff for AND/XOR tests
    for (int i = 0; i < numData; i++)
        hOData[i] = 0;
    hOData[8]  = 0xff;
    hOData[10] = 0xff;

    // ------------------------------------------------------------------
    // 4. Create non-blocking stream + allocate device memory
    // ------------------------------------------------------------------
    //   cudaStreamNonBlocking: this stream does NOT implicitly sync with
    //   the default (NULL) stream. Allows true async overlap.
    // ------------------------------------------------------------------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int *dOData = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dOData, memSize));

    // Copy initial data to device (async)
    CUDA_CHECK(cudaMemcpyAsync(dOData, hOData, memSize,
                               cudaMemcpyHostToDevice, stream));

    // ------------------------------------------------------------------
    // 5. Warm-up + timed kernel launch
    // ------------------------------------------------------------------
    testKernel<<<numBlocks, numThreads, 0, stream>>>(dOData);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Re-init device data for timed run
    CUDA_CHECK(cudaMemcpyAsync(dOData, hOData, memSize,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    testKernel<<<numBlocks, numThreads, 0, stream>>>(dOData);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // ------------------------------------------------------------------
    // 6. Copy result back + verify
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(hOData, dOData, memSize,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("Kernel time: %.4f ms\n", kernel_ms);

    printf("\n--- Results ---\n");
    printf("  [0] atomicAdd  = %d  (expected %d)\n",
           hOData[0], totalThreads * 10);
    printf("  [1] atomicSub  = %d  (expected %d)\n",
           hOData[1], -totalThreads * 10);
    printf("  [2] atomicExch = %d  (any tid in [0, %d))\n",
           hOData[2], totalThreads);
    printf("  [3] atomicMax  = %d  (expected %d)\n",
           hOData[3], totalThreads - 1);
    printf("  [4] atomicMin  = %d  (expected 0)\n",
           hOData[4]);
    printf("  [5] atomicInc  = %d  (expected %d mod 18 = %d)\n",
           hOData[5], totalThreads, totalThreads % 18);
    printf("  [6] atomicDec  = %d\n", hOData[6]);
    printf("  [7] atomicCAS  = %d  (any tid)\n", hOData[7]);
    printf("  [8] atomicAnd  = %d\n", hOData[8]);
    printf("  [9] atomicOr   = %d\n", hOData[9]);
    printf("  [10]atomicXor  = %d  (expected 0xff = 255)\n", hOData[10]);

    // ------------------------------------------------------------------
    // 7. Verify
    // ------------------------------------------------------------------
    printf("\n--- Verification ---\n");
    bool passed = computeGold(hOData, totalThreads);
    printf("  Result: %s\n", passed ? "PASS" : "FAIL");

    // ------------------------------------------------------------------
    // 8. Cleanup
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaFreeHost(hOData));
    CUDA_CHECK(cudaFree(dOData));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nDone!\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}