/**
 * simpleVoteIntrinsics_clean.cu
 *
 * Single-file version of NVIDIA's simpleVoteIntrinsics sample.
 *
 * Core concept -- Warp-level vote intrinsics:
 *   Within a single warp (32 threads), threads can "vote" on a predicate.
 *   The result is IDENTICAL across all participating threads -- it's a
 *   warp-level collective, not a per-thread operation.
 *
 *   Two fundamental vote functions:
 *     __any_sync(mask, predicate) -- returns non-zero if ANY thread
 *         in the warp has a non-zero predicate.
 *     __all_sync(mask, predicate) -- returns non-zero only if ALL
 *         threads in the warp have a non-zero predicate.
 *
 *   Both use a participation mask (0xFFFFFFFF = all 32 threads).
 *   These are essential for warp-level conditional branching -- e.g.
 *   "did any thread detect a target?" or "do all threads agree?"
 *
 * This program:
 *   Test 1: __any_sync
 *   Test 2: __all_sync
 *   Test 3: combined any/all with per-thread predicates across 3 warps
 *
 * Test pattern (4 warps x 32 threads = 128 threads total):
 *   warp 0: all zeros        -> __any=0, __all=0
 *   warp 1: mixed (half 1)   -> __any=1, __all=0
 *   warp 2: mixed (half 1)   -> __any=1, __all=0
 *   warp 3: all ones         -> __any=1, __all=1
 */

#include <cstdio>
#include <cstdlib>
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

// ===========================================================================
// Kernel 1: __any_sync -- returns non-zero if ANY thread's input is non-zero
// ===========================================================================
//   Every thread in the warp gets the SAME result. If even one thread has
//   a non-zero input, all 32 threads in that warp see result=1.
// ===========================================================================
__global__ void VoteAnyKernel1(const unsigned int *input, unsigned int *result,
                                int size)
{
    int tx = threadIdx.x;
    unsigned int mask = 0xffffffff;
    result[tx] = __any_sync(mask, input[tx]);
}

// ===========================================================================
// Kernel 2: __all_sync -- non-zero only if ALL threads' inputs are non-zero
// ===========================================================================
//   Every thread gets the SAME result. Only if ALL 32 threads have non-zero
//   input does every thread see result=1.
// ===========================================================================
__global__ void VoteAllKernel2(const unsigned int *input, unsigned int *result,
                                int size)
{
    int tx = threadIdx.x;
    unsigned int mask = 0xffffffff;
    result[tx] = __all_sync(mask, input[tx]);
}

// ===========================================================================
// Kernel 3: combined any/all + per-thread predicates across 3 warps
// ===========================================================================
//   Launched with 3 warps (96 threads). Each thread writes 3 bools:
//     [0] __any_sync: is ANY thread in my warp at tx >= 48?
//     [1] per-thread:  is MY tx >= 48? (NOT a vote -- per-thread result)
//     [2] __all_sync:  do ALL threads in my warp have tx >= 48?
//
//   Expected results by warp:
//     warp 0 (tx  0-31): [0]=0  [1]=0  [2]=0    (all < 48)
//     warp 1 (tx 32-63): [0]=1  [1]=0/1 mixed   [2]=0  (some >= 48)
//     warp 2 (tx 64-95): [0]=1  [1]=1 everywhere [2]=1  (all >= 48)
// ===========================================================================
__global__ void VoteAnyKernel3(bool *info, int warp_size)
{
    int tx = threadIdx.x;
    unsigned int mask = 0xffffffff;
    bool *offs = info + (tx * 3);    // 3 results per thread

    int half = (warp_size * 3) / 2;  // = 48, boundary between warp 1 and 2

    // [0] __any_sync: any thread in my warp with tx >= 48?
    offs[0] = __any_sync(mask, (tx >= half));

    // [1] Simple per-thread predicate (not a vote -- each thread differs)
    offs[1] = (tx >= half);

    // [2] __all_sync: ALL threads in my warp have tx >= 48?
    if (__all_sync(mask, (tx >= half))) {
        offs[2] = true;
    }
}

// ===========================================================================
// Generate test pattern: 4 warps with different input characteristics
// ===========================================================================
//   numWarps = 4, warpSize = 32, size = 128
//
//   warp 0 [  0.. 31]: all zeros
//   warp 1 [ 32.. 63]: alternating: even index -> 0, odd index -> non-zero
//   warp 2 [ 64.. 95]: alternating: even index -> non-zero, odd index -> 0
//   warp 3 [ 96..127]: all 0xFFFFFFFF
// ===========================================================================
static void genVoteTestPattern(unsigned int *pattern, int size)
{
    int numWarps = 4;
    int warpSize = 32;

    // warp 0: all zeros (size/4 = 32 elements)
    for (int i = 0; i < warpSize; i++)
        pattern[i] = 0x00000000;

    // warp 1: half non-zero (even=0, odd=nonzero)
    for (int i = warpSize; i < 2 * warpSize; i++)
        pattern[i] = (i & 0x01) ? i : 0;

    // warp 2: half non-zero (even=nonzero, odd=0)
    for (int i = 2 * warpSize; i < 3 * warpSize; i++)
        pattern[i] = (i & 0x01) ? 0 : i;

    // warp 3: all ones
    for (int i = 3 * warpSize; i < size; i++)
        pattern[i] = 0xffffffff;
}

// ===========================================================================
// Verification helpers
// ===========================================================================

// Check that all elements in [start, end) are zero
static int checkAllZero(const unsigned int *result, int start, int end,
                         const char *label)
{
    int errors = 0;
    for (int i = start; i < end; i++) {
        if (result[i] != 0) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  %s [%d-%d]: %d/%d FAILED (expected all 0)\n",
               label, start, end - 1, errors, end - start);
    }
    return errors;
}

// Check that all elements in [start, end) are one (non-zero)
static int checkAllOne(const unsigned int *result, int start, int end,
                        const char *label)
{
    int errors = 0;
    for (int i = start; i < end; i++) {
        if (result[i] == 0) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  %s [%d-%d]: %d/%d FAILED (expected all 1)\n",
               label, start, end - 1, errors, end - start);
    }
    return errors;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    const int NUM_WARPS = 4;
    const int WARP_SIZE = 32;
    const int N = NUM_WARPS * WARP_SIZE;  // 128 threads

    int errors1 = 0, errors2 = 0, errors3 = 0;

    printf("[simpleVoteIntrinsics] - Starting...\n\n");

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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: \"%s\", SM %d.%d\n\n",
           prop.name, prop.major, prop.minor);

    // ------------------------------------------------------------------
    // 2. Allocate host + device memory, generate test pattern
    // ------------------------------------------------------------------
    unsigned int *h_input  = new unsigned int[N];
    unsigned int *h_result = new unsigned int[N];

    genVoteTestPattern(h_input, N);

    // Print the test pattern for visual confirmation
    printf("Test pattern (warp layout):\n");
    for (int w = 0; w < NUM_WARPS; w++) {
        printf("  warp %d:  ", w);
        int nonzero = 0;
        for (int i = w * WARP_SIZE; i < (w + 1) * WARP_SIZE; i++) {
            if (h_input[i] != 0) nonzero++;
        }
        if (nonzero == 0)
            printf("all 0");
        else if (nonzero == WARP_SIZE)
            printf("all 1");
        else
            printf("mixed (%d of 32 non-zero)", nonzero);
        printf("\n");
    }
    printf("\n");

    unsigned int *d_input  = nullptr;
    unsigned int *d_result = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input,  N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&d_result, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // ==================================================================
    // Test 1: __any_sync
    // ==================================================================
    printf("[Test 1/3] __any_sync\n");
    printf("  If ANY thread in the warp has non-zero input,\n");
    printf("  ALL threads in that warp return 1.\n\n");

    VoteAnyKernel1<<<1, N>>>(d_input, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    printf("  Expected: warp0=0, warp1=1, warp2=1, warp3=1\n");
    printf("  Got:      ");
    for (int w = 0; w < NUM_WARPS; w++) {
        printf("warp%d=%u  ",
               w, h_result[w * WARP_SIZE]);  // any thread, same result per warp
    }
    printf("\n\n");

    errors1 += checkAllZero(h_result, 0, WARP_SIZE, "__any_sync warp0");
    errors1 += checkAllOne(h_result, WARP_SIZE, 2 * WARP_SIZE, "__any_sync warp1");
    errors1 += checkAllOne(h_result, 2 * WARP_SIZE, 3 * WARP_SIZE, "__any_sync warp2");
    errors1 += checkAllOne(h_result, 3 * WARP_SIZE, N, "__any_sync warp3");
    printf("  Result: %s\n\n", errors1 == 0 ? "PASS" : "FAIL");

    // ==================================================================
    // Test 2: __all_sync
    // ==================================================================
    printf("[Test 2/3] __all_sync\n");
    printf("  Only if ALL threads in the warp have non-zero input,\n");
    printf("  do ALL threads return 1.\n\n");

    VoteAllKernel2<<<1, N>>>(d_input, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    printf("  Expected: warp0=0, warp1=0, warp2=0, warp3=1\n");
    printf("  Got:      ");
    for (int w = 0; w < NUM_WARPS; w++) {
        printf("warp%d=%u  ", w, h_result[w * WARP_SIZE]);
    }
    printf("\n\n");

    errors2 += checkAllZero(h_result, 0, WARP_SIZE, "__all_sync warp0");
    errors2 += checkAllZero(h_result, WARP_SIZE, 2 * WARP_SIZE, "__all_sync warp1");
    errors2 += checkAllZero(h_result, 2 * WARP_SIZE, 3 * WARP_SIZE, "__all_sync warp2");
    errors2 += checkAllOne(h_result, 3 * WARP_SIZE, N, "__all_sync warp3");
    printf("  Result: %s\n\n", errors2 == 0 ? "PASS" : "FAIL");

    // Free test 1/2 resources
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
    delete[] h_input;
    delete[] h_result;

    // ==================================================================
    // Test 3: combined __any_sync, predicate, __all_sync across 3 warps
    // ==================================================================
    printf("[Test 3/3] Combined __any_sync + predicate + __all_sync\n");
    printf("  3 warps (96 threads), 3 results per thread.\n\n");

    const int N3 = WARP_SIZE * 3;         // 96 threads in 3 warps
    const int RESULT_SIZE = N3 * 3;       // 3 bools per thread

    bool *h_info = new bool[RESULT_SIZE]();  // zero-initialized

    bool *d_info = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_info, RESULT_SIZE * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_info, h_info, RESULT_SIZE * sizeof(bool),
                          cudaMemcpyHostToDevice));

    VoteAnyKernel3<<<1, N3>>>(d_info, WARP_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_info, d_info, RESULT_SIZE * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    // Manually verify: check per-thread and per-warp results
    int half = (WARP_SIZE * 3) / 2;  // 48
    for (int tx = 0; tx < N3; tx++) {
        bool v_any = h_info[tx * 3 + 0];      // __any_sync(tx >= 48)
        bool v_pred = h_info[tx * 3 + 1];     // tx >= 48
        bool v_all = h_info[tx * 3 + 2];      // __all_sync(tx >= 48)

        // Verify per-thread predicate
        if (v_pred != (tx >= half)) {
            printf("  FAIL [tx=%d]: predicate expected %d, got %d\n",
                   tx, (int)(tx >= half), (int)v_pred);
            errors3++;
        }

        // Verify __any_sync result
        int warp_id = tx / WARP_SIZE;
        bool expected_any = (warp_id >= 2) || (warp_id == 1 && tx >= half);
        // warp 0: all < 48 → any = false
        // warp 1: some >= 48 (tx 48-63) → any = true
        // warp 2: all >= 48 → any = true
        if (v_any != expected_any) {
            printf("  FAIL [tx=%d]: __any_sync expected %d, got %d\n",
                   tx, (int)expected_any, (int)v_any);
            errors3++;
        }

        // Verify __all_sync result
        bool expected_all = (warp_id >= 2);  // only warp 2: all tx >= 48
        if (v_all != expected_all) {
            printf("  FAIL [tx=%d]: __all_sync expected %d, got %d\n",
                   tx, (int)expected_all, (int)v_all);
            errors3++;
        }
    }

    printf("  Result: %s\n", errors3 == 0 ? "PASS" : "FAIL");

    // Print summary per warp
    printf("\n  Per-warp summary:\n");
    for (int w = 0; w < 3; w++) {
        int base = w * WARP_SIZE;
        printf("  warp %d (tx %d-%d):\n", w, base, base + WARP_SIZE - 1);
        printf("    __any_sync(tx>=48) = %d\n",
               (int)h_info[base * 3 + 0]);  // all same in warp
        printf("    tx >= 48            = %d..%d  (per-thread)\n",
               (int)h_info[base * 3 + 1],
               (int)h_info[(base + WARP_SIZE - 1) * 3 + 1]);
        printf("    __all_sync(tx>=48) = %d\n",
               (int)h_info[base * 3 + 2]);  // all same in warp
    }

    // Cleanup
    delete[] h_info;
    CUDA_CHECK(cudaFree(d_info));

    // ==================================================================
    // Final results
    // ==================================================================
    int totalErrors = errors1 + errors2 + errors3;
    printf("\n========================================\n");
    printf("Total: %d errors (%s)\n",
           totalErrors, totalErrors == 0 ? "ALL PASSED" : "FAILURES FOUND");

    return totalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
