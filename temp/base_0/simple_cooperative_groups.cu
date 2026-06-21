/*
nvcc -std=c++17 -o simple_cooperative_groups simple_cooperative_groups.cu
simple_cooperative_groups.exe

nvcc -o demo1 demo1.cu
demo1.exe
*/

/**
 * Single-file version of NVIDIA's simpleCooperativeGroups sample.
 *
 * Core concept -- Cooperative Groups:
 *   A CUDA 9+ C++ API that abstracts thread cooperation into typed "groups".
 *   Instead of writing raw __syncthreads() and managing shared memory
 *   offsets manually, you create group objects and call their methods.
 *
 *   Group types demonstrated:
 *     thread_block           -- all threads in a block
 *     thread_block_tile<N>   -- a partition of N threads within the block
 *                                (N must be a power of 2, <= warpSize)
 *
 *   Key group methods:
 *     group.size()        -- number of threads in the group
 *     group.thread_rank() -- this thread's index within the group
 *     group.sync()        -- barrier: all threads in group must arrive
 *
 *   The reduction kernel (sumReduction) works with ANY group type --
 *   thread_block, tiled_partition, or even a single warp. This is the
 *   key advantage: write once, compose at different granularities.
 *
 * Reduction algorithm (tree-based):
 *   Round 0: threads 0-31 pair, sum neighbor at distance 16
 *   Round 1: threads 0-15 pair, sum neighbor at distance 8
 *   Round 2: threads 0-7 pair,  sum neighbor at distance 4
 *   Round 3: threads 0-3 pair,  sum neighbor at distance 2
 *   Round 4: threads 0-1 pair,  sum neighbor at distance 1
 *   Final:   thread 0 holds the sum of all 32 values
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// ---------------------------------------------------------------------------
// Error-check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "[ERROR] %s:%d  %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

namespace cg = cooperative_groups;

// ===========================================================================
// Generic sum reduction -- works with ANY cooperative group type
// ===========================================================================
//   Algorithm: tree-based parallel reduction within a group.
//
//   Round 0: distance = size/2 (=16), threads 0-15 add value from partner
//            at thread[lane+16]
//   Round 1: distance = 8, threads 0-7 add partner at thread[lane+8]
//   ...
//   Round 4: distance = 1, thread 0 adds partner at thread[1]
//
//   After all rounds, thread 0 holds the sum of ALL values in the group.
//
//   x: shared memory workspace, must hold at least g.size() ints.
//   val: per-thread contribution to the sum.
//
//   Returns: total sum (thread 0), -1 (all other threads).
// ===========================================================================
__device__ int sumReduction(cg::thread_group g, int *x, int val)
{
    int lane = g.thread_rank();

    for (int i = g.size() / 2; i > 0; i /= 2) {
        // Each thread stores its current value into shared memory
        x[lane] = val;

        // Barrier: ensure ALL threads' values are visible before reading
        g.sync();

        // Active (low-indexed) threads add their partner's value
        if (lane < i)
            val += x[lane + i];

        // Barrier: ensure reads complete before next round's writes
        g.sync();
    }

    // Only the first thread in the group returns the result
    if (g.thread_rank() == 0)
        return val;
    else
        return -1;
}

// ===========================================================================
// Kernel: demonstrates Cooperative Groups at two granularities
// ===========================================================================
__global__ void cgkernel()
{
    // ---- Group 1: whole block ----
    //
    // this_thread_block() returns a thread_block group representing
    // ALL threads in this block.
    //
    cg::thread_block block = cg::this_thread_block();
    int blockSize = block.size();

    // Dynamic shared memory: one int per thread for reduction workspace
    extern __shared__ int workspace[];

    int input, output, expectedOutput;

    // Each thread's contribution to the sum = its rank within the group
    input = block.thread_rank();

    // Analytical result: sum(0..n-1) = (n-1)*n/2
    expectedOutput = (blockSize - 1) * blockSize / 2;

    // Reduce: compute sum of all ranks across the whole block
    output = sumReduction(block, workspace, input);

    // Thread 0 prints the result
    if (block.thread_rank() == 0) {
        printf("Sum of all ranks 0..%d in thread_block: %d (expected %d)\n\n",
               blockSize - 1, output, expectedOutput);

        int numTiles = blockSize / 16;
        printf("Now partitioning the block into %d tile groups "
               "of 16 threads each:\n\n", numTiles);
    }

    block.sync();  // wait for thread 0 to finish printing

    // ---- Group 2: tiled partition of 16 threads each ----
    //
    // tiled_partition<16> splits the block into groups of 16 consecutive
    // threads. In a 64-thread block, this creates 4 groups:
    //   tile 0: threads  0..15
    //   tile 1: threads 16..31
    //   tile 2: threads 32..47
    //   tile 3: threads 48..63
    //
    // NOTE: This is NOT the same as deprecated tiled_partition --
    //       Cooperative Groups version is type-safe and supports
    //       all the same methods as thread_block.
    //
    cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);

    // Each tile gets its own region in the shared memory workspace.
    // Since tiles are non-overlapping partitions of the block,
    // each tile can use a dedicated slice of the workspace array.
    // Offset = (thread's block-level rank) - (thread's tile-level rank)
    int workspaceOffset = block.thread_rank() - tile16.thread_rank();

    // Now each thread contributes its tile-level rank
    input = tile16.thread_rank();

    // Sum of ranks 0..15 = 15*16/2 = 120
    expectedOutput = 15 * 16 / 2;

    // Reduce within each tile group independently
    output = sumReduction(tile16, workspace + workspaceOffset, input);

    // Thread 0 in each tile prints its tile's result
    if (tile16.thread_rank() == 0) {
        printf("  Tile group (offset=%3d): sum of ranks 0..15 = %d "
               "(expected %d)\n",
               workspaceOffset, output, expectedOutput);
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[simpleCooperativeGroups] - Starting...\n\n");

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
    // 2. Launch: 1 block x 64 threads, 64 ints of shared memory
    // ------------------------------------------------------------------
    int blocksPerGrid   = 1;
    int threadsPerBlock = 64;
    int smemBytes       = threadsPerBlock * (int)sizeof(int);

    printf("Launching 1 block with %d threads, %d bytes shared memory...\n\n",
           threadsPerBlock, smemBytes);

    cgkernel<<<blocksPerGrid, threadsPerBlock, smemBytes>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    printf("\nDone!\n");
    return EXIT_SUCCESS;
}
