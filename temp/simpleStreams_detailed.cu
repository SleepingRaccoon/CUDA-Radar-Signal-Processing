/**
 * simpleStreams_detailed.cu
 *
 * A self-contained, detailed version of NVIDIA's simpleStreams sample.
 * No external helper libraries -- all code is inlined and rewritten cleanly.
 *
 * Retained technical details:
 *   - SM version -> cores/SM lookup table (Fermi through Blackwell+)
 *   - scale_factor auto-scaling by GPU compute capability
 *   - Two host-memory allocation strategies:
 *       A) VirtualAlloc/mmap + cudaHostRegister ("generic pinning")
 *       B) cudaMallocHost (direct pinned allocation)
 *   - cudaDeviceBlockingSync flag for low-CPU-usage event waits
 *   - Baseline (single memcpy + single kernel) vs non-streamed vs streamed
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// On Windows we need the VirtualAlloc / VirtualFree API.
// _WIN32 is defined for both 32-bit and 64-bit Windows builds.
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>   // mmap / munmap
#endif

// ---------------------------------------------------------------------------
// Error check macro
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

#define MEMORY_ALIGNMENT 4096
// Round a pointer up to the next page boundary
#define ALIGN_UP(ptr, alignment)                                               \
    ((void *)((((size_t)(ptr)) + ((alignment)-1)) & ~((alignment)-1)))

// ---------------------------------------------------------------------------
// SM version -> cores per SM lookup
// ---------------------------------------------------------------------------
//   Each entry: SM major/minor packed into a single byte pair << 4.
//   e.g. SM 8.0 = 0x80, SM 8.9 = 0x89, SM 9.0 = 0x90
// ---------------------------------------------------------------------------
static int sm_cores_per_sm(int major, int minor)
{
    // Compact table: (major << 4) | minor  ->  cores per SM
    struct { int sm; int cores; } table[] = {
        {0x30, 192},   // Fermi
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},   // Kepler
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},   // Maxwell
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},   // Volta
        {0x72,  64},
        {0x75,  64},   // Turing
        {0x80,  64},   // Ampere A100
        {0x86, 128},   // Ampere GA10x
        {0x87, 128},
        {0x89, 128},   // Ada Lovelace
        {0x90, 128},   // Hopper
        {0xa0, 128},   // Blackwell
        {0xa1, 128},
        {0xa3, 128},
        {0xb0, 128},
        {0xc0, 128},
        {0xc1, 128},
    };
    int n = sizeof(table) / sizeof(table[0]);
    int encoded = (major << 4) | minor;

    for (int i = 0; i < n; i++) {
        if (table[i].sm == encoded)
            return table[i].cores;
    }

    // Unknown SM: fall back to the last entry
    printf("[WARNING] SM %d.%d not found, assuming %d cores/SM\n",
           major, minor, table[n - 1].cores);
    return table[n - 1].cores;
}

// ---------------------------------------------------------------------------
// Kernel: add 'factor' to each element, 'num_iterations' times
// ---------------------------------------------------------------------------
//   Larger num_iterations -> heavier kernel -> more visible stream overlap.
//   Non-coalesced on purpose to burn more compute cycles.
// ---------------------------------------------------------------------------
__global__ void init_array(int *g_data, int factor, int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < num_iterations; i++) {
        g_data[idx] += factor;
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------
static bool check_result(const int *data, int n, int expected)
{
    for (int i = 0; i < n; i++) {
        if (data[i] != expected) {
            printf("  FAIL at [%d]: got %d, expected %d\n",
                   i, data[i], expected);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Host-memory allocation (two strategies)
// ---------------------------------------------------------------------------
//   Strategy A (bPinGenericMemory == true):
//     OS-level page-aligned alloc (VirtualAlloc / mmap) + cudaHostRegister
//     to pin it.  Advantage: can pin any existing memory, not just new allocs.
//
//   Strategy B (bPinGenericMemory == false):
//     cudaMallocHost -- CUDA driver allocates pinned memory directly.
//     Simpler, but can only allocate fresh memory (no retroactive pinning).
//
//   On return: *pp_a       = base address (for free)
//              *pp_aligned = page-aligned pointer (for actual use)
// ---------------------------------------------------------------------------
static void alloc_host_pinned(bool use_generic,
                               int **pp_base, int **pp_aligned, int nbytes)
{
    if (use_generic) {
        // --- Strategy A ---
#ifdef _WIN32
        printf("  VirtualAlloc()  %.2f MB  (generic page-aligned)\n",
               (float)nbytes / 1048576.0f);
        *pp_base = (int *)VirtualAlloc(NULL, nbytes + MEMORY_ALIGNMENT,
                                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        printf("  mmap()  %.2f MB  (generic page-aligned)\n",
               (float)nbytes / 1048576.0f);
        *pp_base = (int *)mmap(NULL, nbytes + MEMORY_ALIGNMENT,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
        *pp_aligned = (int *)ALIGN_UP(*pp_base, MEMORY_ALIGNMENT);

        printf("  cudaHostRegister() pinning  %.2f MB\n",
               (float)nbytes / 1048576.0f);
        CUDA_CHECK(cudaHostRegister(*pp_aligned, nbytes, cudaHostRegisterMapped));
    } else {
        // --- Strategy B ---
        printf("  cudaMallocHost()  %.2f MB  (pinned)\n",
               (float)nbytes / 1048576.0f);
        CUDA_CHECK(cudaMallocHost((void **)pp_base, nbytes));
        *pp_aligned = *pp_base;
    }
}

static void free_host_pinned(bool use_generic,
                              int **pp_base, int **pp_aligned, int nbytes)
{
    if (use_generic) {
        CUDA_CHECK(cudaHostUnregister(*pp_aligned));
#ifdef _WIN32
        VirtualFree(*pp_base, 0, MEM_RELEASE);
#else
        munmap(*pp_base, nbytes + MEMORY_ALIGNMENT);
#endif
    } else {
        CUDA_CHECK(cudaFreeHost(*pp_base));
    }
    *pp_base    = NULL;
    *pp_aligned = NULL;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    // ------------------------------------------------------------------
    // 1. Parameters
    // ------------------------------------------------------------------
    int cuda_device    = 0;
    int nstreams       = 4;
    int nreps          = 10;
    int n              = 16 * 1024 * 1024;   // 16M ints = 64 MB
    int nbytes         = n * sizeof(int);
    int niterations    = 5;
    int factor         = 5;

    dim3 threads(512, 1);
    dim3 blocks;

    float time_memcpy, time_kernel, elapsed_time, time_nostream;
    float scale_factor = 1.0f;

    // Which memory allocation strategy?
    // Mac does NOT support generic page-aligned pinning.
#if defined(__APPLE__) || defined(MACOSX)
    bool use_generic_pin = false;
#else
    bool use_generic_pin = true;
#endif

    // Use cudaEventBlockingSync so CPU yields while waiting (lower CPU usage)
    int sync_method = cudaDeviceBlockingSync;

    printf("========================================\n"
           "[ simpleStreams  --  CUDA Streams Demo ]\n"
           "========================================\n");

    // ------------------------------------------------------------------
    // 2. Select GPU and query hardware
    // ------------------------------------------------------------------
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        printf("ERROR: no CUDA-capable device found\n");
        return 2;
    }
    if (cuda_device >= ndev) {
        printf("ERROR: device %d out of range (0-%d)\n", cuda_device, ndev - 1);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(cuda_device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cuda_device));

    if (use_generic_pin) {
        printf("Device \"%s\" canMapHostMemory: %s\n",
               prop.name, prop.canMapHostMemory ? "Yes" : "No");
        if (!prop.canMapHostMemory) {
            printf("  -> falling back to cudaMallocHost\n");
            use_generic_pin = false;
        }
    }

    // ------------------------------------------------------------------
    // 3. scale_factor: adjust workload to GPU strength
    // ------------------------------------------------------------------
    //   Baseline is 32 "cores" (old Fermi with 1 SM).
    //   For a stronger GPU, reduce array size so the test finishes
    //   in reasonable time.  For weaker GPUs, size stays unchanged
    //   (scale_factor >= 1).
    // ------------------------------------------------------------------
    int total_cores = sm_cores_per_sm(prop.major, prop.minor)
                      * prop.multiProcessorCount;
    scale_factor = (float)fmax(32.0f / (float)total_cores, 1.0f);
    n      = (int)roundf((float)n / scale_factor);
    nbytes = n * sizeof(int);

    printf("\n> SM %d.%d  |  %d SMs x %d cores = %d CUDA cores\n",
           prop.major, prop.minor,
           prop.multiProcessorCount,
           sm_cores_per_sm(prop.major, prop.minor),
           total_cores);
    printf("> scale_factor = %.4f  (data shrunk by 1/%.4f = %.2f%%)\n",
           1.0f / scale_factor, scale_factor,
           (1.0f - 1.0f / scale_factor) * 100.0f);
    printf("> array size   = %d ints  (%d MB)\n\n", n, nbytes / (1024 * 1024));

    // ------------------------------------------------------------------
    // 4. Device flags: blocking sync + (optional) mapped host memory
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaSetDeviceFlags(
        sync_method | (use_generic_pin ? cudaDeviceMapHost : 0)));

    // ------------------------------------------------------------------
    // 5. Allocate memory
    // ------------------------------------------------------------------
    int *h_base    = NULL;
    int *h_aligned = NULL;
    int *d_a       = NULL;

    alloc_host_pinned(use_generic_pin, &h_base, &h_aligned, nbytes);
    CUDA_CHECK(cudaMalloc((void **)&d_a, nbytes));
    CUDA_CHECK(cudaMemset(d_a, 0, nbytes));

    blocks.x = n / threads.x;

    // ------------------------------------------------------------------
    // 6. Create streams
    // ------------------------------------------------------------------
    cudaStream_t *streams = (cudaStream_t *)malloc(
                                nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // ------------------------------------------------------------------
    // 7. Create events for timing
    // ------------------------------------------------------------------
    int event_flags = (sync_method == cudaDeviceBlockingSync)
                      ? cudaEventBlockingSync : cudaEventDefault;
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_start, event_flags));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_stop, event_flags));

    // ==================================================================
    // 8. Baseline: single memcpy + single kernel
    // ==================================================================
    printf("--- Baseline ---\n");

    // 8a. Single D2H memcpy
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_aligned, d_a, nbytes,
                               cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_memcpy, ev_start, ev_stop));
    printf("  Single memcpy (D->H):  %.2f ms\n", time_memcpy);

    // 8b. Single kernel (includes driver-init + GPU warm-up)
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    init_array<<<blocks, threads, 0, streams[0]>>>(d_a, factor, niterations);
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, ev_start, ev_stop));
    printf("  Single kernel:         %.2f ms\n", time_kernel);
    printf("    ^ first kernel includes driver init + GPU warm-up\n");
    printf("      steady-state is much faster (see methods below)\n");

    // ==================================================================
    // 9. Without streams: fully serial
    // ==================================================================
    printf("\n--- Method 1: Without Streams (serial) ---\n");

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int k = 0; k < nreps; k++) {
        init_array<<<blocks, threads>>>(d_a, factor, niterations);
        CUDA_CHECK(cudaMemcpy(h_aligned, d_a, nbytes,
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, ev_start, ev_stop));
    time_nostream = elapsed_time;
    printf("  Total: %.2f ms  |  per round: %.2f ms\n",
           elapsed_time, elapsed_time / nreps);

    // ==================================================================
    // 10. With N streams: overlapping compute + transfer
    // ==================================================================
    printf("\n--- Method 2: With %d Streams (overlapping) ---\n", nstreams);

    memset(h_aligned, 0xFF, nbytes);
    CUDA_CHECK(cudaMemset(d_a, 0, nbytes));

    // Each stream handles 1/nstreams of the data
    blocks.x = n / (nstreams * threads.x);

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    for (int k = 0; k < nreps; k++) {
        // All kernels first, one per stream
        for (int i = 0; i < nstreams; i++) {
            init_array<<<blocks, threads, 0, streams[i]>>>(
                d_a + i * n / nstreams, factor, niterations);
        }
        // Then all async memcpies (each waits only on its own stream)
        for (int i = 0; i < nstreams; i++) {
            CUDA_CHECK(cudaMemcpyAsync(
                h_aligned + i * n / nstreams,
                d_a       + i * n / nstreams,
                nbytes / nstreams,
                cudaMemcpyDeviceToHost,
                streams[i]));
        }
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, ev_start, ev_stop));
    printf("  Total: %.2f ms  |  per round: %.2f ms\n",
           elapsed_time, elapsed_time / nreps);

    // ==================================================================
    // 11. Comparison
    // ==================================================================
    printf("\n========================================\n"
           "Comparison Summary\n"
           "========================================\n");
    printf("  Without Streams:  %.2f ms/round  (%d rounds)\n",
           time_nostream / nreps, nreps);
    printf("  With %d Streams:  %.2f ms/round  (%d rounds)\n",
           nstreams, elapsed_time / nreps, nreps);
    printf("\n");
    printf("  Speedup depends on kernel/memcpy time ratio:\n");
    printf("    kernel > memcpy  -> good overlap  (compute bound)\n");
    printf("    kernel < memcpy  -> PCIe bandwidth bottleneck\n");
    printf("  Adjust niterations (currently %d) to control kernel time\n",
           niterations);

    // ==================================================================
    // 12. Verification
    // ==================================================================
    printf("\n--- Verification ---\n");
    int expected = factor * nreps * niterations;
    bool pass = check_result(h_aligned, n, expected);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    // ==================================================================
    // 13. Cleanup
    // ==================================================================
    printf("\n--- Cleanup ---\n");
    for (int i = 0; i < nstreams; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    free(streams);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free_host_pinned(use_generic_pin, &h_base, &h_aligned, nbytes);
    CUDA_CHECK(cudaFree(d_a));

    printf("Done!\n");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
