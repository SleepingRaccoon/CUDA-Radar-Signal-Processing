/**
 * simpleStreams_clean.cu
 *
 * A clean, minimal version of NVIDIA's simpleStreams sample.
 *
 * Core concept:
 *   GPU can do two things simultaneously -- run kernels and transfer data
 *   (Device <-> Host), as long as they are in different CUDA Streams.
 *   This is called "compute and transfer overlap".
 *
 * This program compares two approaches:
 *   (1) Without Streams: kernel -> wait -> memcpy -> wait -> next round
 *   (2) With 4 Streams: split data into 4 chunks, each chunk runs its
 *       own kernel -> memcpy pipeline independently in its own stream.
 *       The GPU scheduler overlaps operations across streams automatically.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// A kernel that simulates some real work
// ---------------------------------------------------------------------------
//   Each thread repeatedly adds to its element.
//   Larger num_iterations -> longer kernel time -> more visible overlap.
// ---------------------------------------------------------------------------
__global__ void init_array(int *g_data, int factor, int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Non-coalesced access pattern on purpose: burns more compute time
    for (int i = 0; i < num_iterations; i++) {
        g_data[idx] += factor;
    }
}

int main()
{
    // ======================================================================
    // 1. Parameters
    // ======================================================================
    int nstreams      = 4;                 // number of streams to use
    int nreps         = 10;                // repetitions per experiment
    int n             = 16 * 1024 * 1024;  // array size (16M ints)
    int nbytes        = n * sizeof(int);   // ~64 MB
    int niterations   = 5;                 // kernel inner-loop count
    int factor        = 5;                 // per-iteration increment

    dim3 threads(512, 1);                  // 512 threads per block
    dim3 blocks(n / threads.x, 1);         // total blocks

    float time_memcpy, time_kernel;        // single-operation timings
    float elapsed_time;                    // multi-round total time
    float time_nostream;                   // non-streamed total (saved separately)

    printf("========================================\n");
    printf("CUDA Streams Learning Example\n");
    printf("========================================\n");
    printf("Data size : %d ints (%d MB)\n", n, nbytes / 1024 / 1024);
    printf("Streams   : %d\n", nstreams);
    printf("Repetitions: %d\n\n", nreps);

    // ======================================================================
    // 2. Allocate memory
    // ======================================================================
    //   Host memory must use cudaMallocHost (pinned memory).
    //   Without pinned memory, cudaMemcpyAsync silently falls back to
    //   synchronous behavior -- no overlap possible.
    // ======================================================================
    int *h_a = NULL;   // host (pinned)
    int *d_a = NULL;   // device

    cudaMallocHost((void **)&h_a, nbytes);
    cudaMalloc((void **)&d_a, nbytes);
    cudaMemset(d_a, 0x00, nbytes);

    printf("Host   memory (pinned): %d MB\n", nbytes / 1024 / 1024);
    printf("Device memory         : %d MB\n\n", nbytes / 1024 / 1024);

    // ======================================================================
    // 3. Create CUDA Streams
    // ======================================================================
    cudaStream_t streams[4];
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // ======================================================================
    // 4. Create CUDA Events (for GPU-side accurate timing)
    // ======================================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ======================================================================
    // 5. Baseline: measure single memcpy and single kernel duration
    // ======================================================================
    printf("--- Baseline: single-operation timing ---\n");

    // 5a. Single D2H memcpy
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memcpy, start, stop);
    printf("  Single memcpy (D->H): %6.2f ms\n", time_memcpy);

    // 5b. Single kernel
    cudaEventRecord(start, 0);
    init_array<<<blocks, threads, 0, streams[0]>>>(d_a, factor, niterations);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel, start, stop);
    printf("  Single kernel:        %6.2f ms\n", time_kernel);
    printf("    ^ First kernel includes driver init + GPU warm-up.\n");
    printf("      Steady-state kernel is much faster (see method 1/2).\n");
    printf("  (Adjust niterations to control kernel duration)\n\n");

    // ======================================================================
    // 6. Without Streams: fully serial execution
    // ======================================================================
    //   Each round: launch kernel -> wait for it -> memcpy -> wait for it.
    //   Simple, but the GPU compute units and data bus never work at the
    //   same time, leading to low utilization.
    // ======================================================================
    printf("--- Method 1: Without Streams (serial) ---\n");

    cudaEventRecord(start, 0);

    for (int k = 0; k < nreps; k++) {
        // Run kernel: add factor to every element, niterations times
        init_array<<<blocks, threads>>>(d_a, factor, niterations);

        // Synchronous memcpy: blocks until copy finishes
        // (cudaMemcpy is blocking -- CPU waits for completion)
        cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    time_nostream = elapsed_time;   // save before it gets overwritten below
    printf("  Total: %.2f ms\n", elapsed_time);
    printf("  Per round: %.2f ms\n\n", elapsed_time / nreps);

    // ======================================================================
    // 7. With Streams: overlapping compute and data transfer
    // ======================================================================
    //   Split data into nstreams chunks, each chunk runs independently
    //   in its own stream.
    //
    //   Key difference: launch ALL kernels first (one per stream),
    //   then launch ALL async memcpies.
    //
    //   The GPU scheduler interleaves different streams' operations:
    //
    //     stream 0: kernel --> memcpy D2H
    //     stream 1:        kernel --> memcpy D2H
    //     stream 2:               kernel --> memcpy D2H
    //     stream 3:                      kernel --> memcpy D2H
    //                                    ^ time axis
    //
    //   While stream 0 copies data, stream 1/2/3 kernels can run.
    //   Total time ~ max(total compute, total transfer), not sum.
    // ======================================================================
    printf("--- Method 2: With %d Streams (overlapping) ---\n", nstreams);

    // Reset data: device = all 0s, host = all 0xFFs for verification
    cudaMemset(d_a, 0x00, nbytes);
    memset(h_a, 0xFF, nbytes);

    cudaEventRecord(start, 0);

    for (int k = 0; k < nreps; k++) {
        // Step 1: launch kernel in each stream, each working on its chunk
        for (int i = 0; i < nstreams; i++) {
            int offset = i * n / nstreams;
            int chunk_blocks = blocks.x / nstreams;
            init_array<<<chunk_blocks, threads, 0, streams[i]>>>(
                d_a + offset, factor, niterations);
        }

        // Step 2: launch async memcpy in each stream
        //   stream i's memcpy waits for stream i's kernel,
        //   but does NOT wait for any other stream's work.
        for (int i = 0; i < nstreams; i++) {
            int offset     = i * n / nstreams;
            int chunk_size = nbytes / nstreams;
            cudaMemcpyAsync(h_a + offset, d_a + offset, chunk_size,
                            cudaMemcpyDeviceToHost, streams[i]);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("  Total: %.2f ms\n", elapsed_time);
    printf("  Per round: %.2f ms\n\n", elapsed_time / nreps);

    // ======================================================================
    // 8. Result comparison
    // ======================================================================
    printf("========================================\n");
    printf("Comparison Summary\n");
    printf("========================================\n");
    printf("  Without Streams per round: ~ %.2f ms  (measured from %d rounds)\n",
           time_nostream / nreps, nreps);
    printf("  With Streams per round:    ~ %.2f ms  (measured from %d rounds)\n",
           elapsed_time / nreps, nreps);
    printf("  Speedup depends on kernel/memcpy time ratio:\n");
    printf("    - kernel > memcpy -> good overlap (compute bound)\n");
    printf("    - kernel < memcpy -> bandwidth bound (PCIe limits)\n\n");

    // ======================================================================
    // 9. Verify correctness
    // ======================================================================
    //   Each element is incremented: factor * nreps * niterations times.
    //   Initial value is 0, so final = factor * nreps * niterations.
    // ======================================================================
    int expected = factor * nreps * niterations;
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_a[i] != expected) {
            printf("FAIL at [%d]: expected %d, got %d\n",
                   i, expected, h_a[i]);
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n\n", correct ? "PASS" : "FAIL");

    // ======================================================================
    // 10. Cleanup
    // ======================================================================
    for (int i = 0; i < nstreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_a);
    cudaFree(d_a);

    printf("Done!\n");
    return 0;
}
