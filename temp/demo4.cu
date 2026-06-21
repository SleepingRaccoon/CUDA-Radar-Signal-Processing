/*
    nvcc -o demo4 demo4.cu
    demo4.exe

    nvcc -o simple_multicopy simple_multicopy.cu
    simple_multicopy.exe
*/

/**
 * simpleMultiCopy_clean.cu
 *
 * Single-file version of NVIDIA's simpleMultiCopy sample.
 *
 * Core concept:
 *   GPUs with asyncEngineCount > 1 (Quadro/Tesla, CC >= 2.0) can overlap
 *   TWO memcopies with kernel execution simultaneously. This sample sets up
 *   a pipelined processing pattern: while stream i runs a kernel on current
 *   data, stream i+1 uploads the next chunk and stream i-1 downloads the
 *   previous result.
 *
 *   Key technique -- ping-pong buffers:
 *     stream 0: [upload 0] -> [kernel 0] -> [download 0]
 *     stream 1:              [upload 1] -> [kernel 1] -> [download 1]
 *     stream 2:                           [upload 2] -> [kernel 2] -> [download 2]
 *     stream 3:                                        [upload 3] -> [kernel 3] -> [download 3]
 *
 *   Each stream works on independent data buffers. Events (cycleDone[i])
 *   ensure that no stream overwrites a buffer still in use by another stream.
 *
 * Uncomment to simulate data source/sink IO times on the CPU side:
 *   #define SIMULATE_IO
 */

// #define SIMULATE_IO

#include <cstdio>
#include <cstdlib>
#include <cmath>

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
// Kernel: increment each element by 1, repeated inner_reps times
// ---------------------------------------------------------------------------
//   inner_reps controls kernel duration -- more reps = heavier compute.
// ---------------------------------------------------------------------------
__global__ void incKernel(int *g_out, const int *g_in, int N, int inner_reps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        for (int i = 0; i < inner_reps; ++i) {
            g_out[idx] = g_in[idx] + 1;
        }
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[simpleMultiCopy] - Starting...\n");

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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("> Using CUDA device [0]: %s\n", prop.name);

    // ------------------------------------------------------------------
    // 2. Parameters
    // ------------------------------------------------------------------
    const int STREAM_COUNT = 4;
    int N          = 1 << 22;       // 4M ints
    int nreps      = 10;            // experiment repetitions
    int inner_reps = 5;             // kernel inner-loop count
    dim3 block(512);
    int   memsize     = N * (int)sizeof(int);
    int   thread_blocks = N / block.x;
    dim3  grid(thread_blocks);

    // ------------------------------------------------------------------
    // 3. Allocate resources
    // ------------------------------------------------------------------
    int *h_data_source = new int[N];
    int *h_data_sink   = new int[N];

    // Per-stream buffers (4 sets: in/out on host + device)
    int   *h_data_in[STREAM_COUNT];
    int   *h_data_out[STREAM_COUNT];
    int   *d_data_in[STREAM_COUNT];
    int   *d_data_out[STREAM_COUNT];
    cudaEvent_t  cycleDone[STREAM_COUNT];
    cudaStream_t stream[STREAM_COUNT];

    for (int i = 0; i < STREAM_COUNT; ++i) {
        // Host memory: use cudaHostAlloc (pinned) so async memcpy works
        CUDA_CHECK(cudaHostAlloc((void **)&h_data_in[i],  memsize, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void **)&h_data_out[i], memsize, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc((void **)&d_data_in[i],  memsize));
        CUDA_CHECK(cudaMalloc((void **)&d_data_out[i], memsize));
        CUDA_CHECK(cudaMemset(d_data_in[i], 0, memsize));

        CUDA_CHECK(cudaStreamCreate(&stream[i]));
        CUDA_CHECK(cudaEventCreate(&cycleDone[i]));

        // Prime the event so the first iteration doesn't wait on garbage
        CUDA_CHECK(cudaEventRecord(cycleDone[i], stream[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ------------------------------------------------------------------
    // 4. Initialize data
    // ------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        h_data_source[i] = 0;
    }
    for (int i = 0; i < STREAM_COUNT; ++i) {
        for (int j = 0; j < N; ++j) {
            h_data_in[i][j] = 0;
        }
    }

    // ------------------------------------------------------------------
    // 5. Kernel warm-up (avoid cold-launch in baseline)
    // ------------------------------------------------------------------
    incKernel<<<grid, block>>>(d_data_out[0], d_data_in[0], N, inner_reps);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // 6. Baseline: measure single-operation times
    // ------------------------------------------------------------------
    float memcpy_h2d_time, memcpy_d2h_time, kernel_time;

    // 6a. Single H2D memcpy
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_data_in[0], h_data_in[0], memsize,
                               cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    // 6b. Single D2H memcpy
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_data_out[0], d_data_out[0], memsize,
                               cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));

    // 6c. Single kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    incKernel<<<grid, block, 0, 0>>>(d_data_out[0], d_data_in[0], N, inner_reps);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    // ------------------------------------------------------------------
    // 7. Device capability info
    // ------------------------------------------------------------------
    printf("\n");
    printf("Relevant properties of this CUDA device\n");

    int canOverlap = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&canOverlap, cudaDevAttrGpuOverlap, 0));
    printf("(%s) Can overlap one CPU<>GPU data transfer with GPU kernel execution\n"
           "    (device property \"cudaDevAttrGpuOverlap\")\n",
           canOverlap ? "X" : " ");

    printf("(%s) Can overlap two CPU<>GPU data transfers with GPU kernel execution\n"
           "    (Compute Capability >= 2.0 AND (Tesla or Quadro K4000+))\n",
           (prop.major >= 2 && prop.asyncEngineCount > 1) ? "X" : " ");

    // ------------------------------------------------------------------
    // 8. Print baseline timings + theoretical limits
    // ------------------------------------------------------------------
    // Bandwidth in GiB/s (1024^3 bytes per GiB)
    double bytes_to_gib = 1.0 / (1024.0 * 1024.0 * 1024.0);

    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %.3f ms (%.3f GiB/s)\n",
           memcpy_h2d_time,
           (memsize * bytes_to_gib) / (memcpy_h2d_time * 1e-3));
    printf(" Memcpy device to host\t: %.3f ms (%.3f GiB/s)\n",
           memcpy_d2h_time,
           (memsize * bytes_to_gib) / (memcpy_d2h_time * 1e-3));
    // Kernel bandwidth: each iteration reads g_in AND writes g_out,
    // so total data moved = inner_reps * 2 * memsize.
    printf(" Kernel\t\t\t: %.3f ms (%.3f GiB/s)\n",
           kernel_time,
           (inner_reps * 2.0 * memsize * bytes_to_gib) / (kernel_time * 1e-3));

    printf("\n");
    printf("Theoretical limits for speedup from overlapped data transfers:\n");
    printf("  No overlap at all (transfer-kernel-transfer): %.3f ms\n",
           memcpy_h2d_time + memcpy_d2h_time + kernel_time);
    // One copy engine (H2D and D2H serial): kernel can overlap with
    // either H2D or D2H, but not both simultaneously. Two possible
    // orderings, pick the better one:
    //   [H2D][kernel // D2H]  or  [kernel // H2D][D2H]
    printf("  Compute overlaps with ONE transfer:           %.3f ms\n",
           fmin(fmax(kernel_time, memcpy_h2d_time) + memcpy_d2h_time,
                memcpy_h2d_time + fmax(kernel_time, memcpy_d2h_time)));
    // Two copy engines: H2D, D2H, and kernel all run in parallel.
    printf("  Compute overlaps with BOTH transfers:         %.3f ms\n",
           fmax(fmax(memcpy_h2d_time, memcpy_d2h_time), kernel_time));

    // ------------------------------------------------------------------
    // 9. Pipelined processing: serial (1 stream) vs overlapped (N streams)
    // ------------------------------------------------------------------
    //   processWithStreams helper (defined below by lambda)
    // ------------------------------------------------------------------
    auto processWithStreams = [&](int streams_used) -> float {
        int current_stream = 0;

        CUDA_CHECK(cudaEventRecord(start, 0));

        for (int i = 0; i < nreps; ++i) {
            int next_stream = (current_stream + 1) % streams_used;

            // Uncomment to simulate data source/sink I/O:
#ifdef SIMULATE_IO
            // Store the result
            memcpy(h_data_sink,   h_data_out[current_stream], memsize);
            // Read new input
            memcpy(h_data_in[next_stream], h_data_source, memsize);
#endif

            // Wait until the next stream's previous cycle is fully done
            CUDA_CHECK(cudaEventSynchronize(cycleDone[next_stream]));

            // Process current frame
            incKernel<<<grid, block, 0, stream[current_stream]>>>(
                d_data_out[current_stream], d_data_in[current_stream],
                N, inner_reps);

            // Upload next frame (overlaps with kernel above)
            CUDA_CHECK(cudaMemcpyAsync(
                d_data_in[next_stream], h_data_in[next_stream],
                memsize, cudaMemcpyHostToDevice, stream[next_stream]));

            // Download current frame (overlaps with both)
            CUDA_CHECK(cudaMemcpyAsync(
                h_data_out[current_stream], d_data_out[current_stream],
                memsize, cudaMemcpyDeviceToHost, stream[current_stream]));

            // Mark this stream's cycle complete
            CUDA_CHECK(cudaEventRecord(cycleDone[current_stream],
                                       stream[current_stream]));

            current_stream = next_stream;
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    };

    // 9a. Run fully serialized (1 stream -- no overlap at all)
    float serial_time = processWithStreams(1);

    // Reset input data for the overlapped run
    for (int i = 0; i < STREAM_COUNT; ++i) {
        CUDA_CHECK(cudaMemset(d_data_in[i], 0, memsize));
        for (int j = 0; j < N; ++j) {
            h_data_in[i][j] = 0;
        }
        // Re-prime events
        CUDA_CHECK(cudaEventRecord(cycleDone[i], stream[i]));
    }

    // 9b. Run overlapped (STREAM_COUNT streams)
    float overlap_time = processWithStreams(STREAM_COUNT);

    // ------------------------------------------------------------------
    // 10. Print results
    // ------------------------------------------------------------------
    printf("\n");
    printf("Average measured timings over %d repetitions:\n", nreps);
    printf("  Fully serialized (1 stream)  : %.3f ms\n", serial_time / nreps);
    printf("  Overlapped (%d streams)       : %.3f ms\n",
           STREAM_COUNT, overlap_time / nreps);
    printf("  Speedup                      : %.2fx\n",
           serial_time / overlap_time);

    printf("\n");
    // Throughput: each round does H2D (upload) + D2H (download),
    // so total data = nreps * 2 * memsize.
    printf("Measured throughput:\n");
    printf("  Fully serialized             : %.3f GiB/s\n",
           (nreps * 2.0 * memsize * bytes_to_gib) / (serial_time * 1e-3));
    printf("  Overlapped (%d streams)      : %.3f GiB/s\n",
           STREAM_COUNT,
           (nreps * 2.0 * memsize * bytes_to_gib) / (overlap_time * 1e-3));

    // ------------------------------------------------------------------
    // 11. Verify: every output element should be 1
    // ------------------------------------------------------------------
    printf("\n--- Verification ---\n");

    // Re-run in serial mode to populate h_data_out for verification
    for (int i = 0; i < STREAM_COUNT; ++i) {
        CUDA_CHECK(cudaMemset(d_data_in[i], 0, memsize));
        for (int j = 0; j < N; ++j) {
            h_data_in[i][j] = 0;
        }
        CUDA_CHECK(cudaEventRecord(cycleDone[i], stream[i]));
    }
    processWithStreams(1);

    bool passed = true;
    for (int j = 0; j < STREAM_COUNT; ++j) {
        for (int i = 0; i < N; ++i) {
            if (h_data_out[j][i] != 1) {
                printf("  FAIL at stream[%d][%d]: got %d, expected 1\n",
                       j, i, h_data_out[j][i]);
                passed = false;
                break;
            }
        }
        if (!passed) break;
    }
    printf("  Result: %s\n", passed ? "PASS" : "FAIL");

    // ------------------------------------------------------------------
    // 12. Cleanup
    // ------------------------------------------------------------------
    delete[] h_data_source;
    delete[] h_data_sink;

    for (int i = 0; i < STREAM_COUNT; ++i) {
        CUDA_CHECK(cudaFreeHost(h_data_in[i]));
        CUDA_CHECK(cudaFreeHost(h_data_out[i]));
        CUDA_CHECK(cudaFree(d_data_in[i]));
        CUDA_CHECK(cudaFree(d_data_out[i]));
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
        CUDA_CHECK(cudaEventDestroy(cycleDone[i]));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nDone!\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
