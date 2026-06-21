/**
 * UnifiedMemoryStreams_clean.cu
 *
 * Single-file version of NVIDIA's UnifiedMemoryStreams sample.
 *
 * Core concepts demonstrated:
 *
 *   1. UNIFIED MEMORY (cudaMallocManaged)
 *      A single pointer works on BOTH CPU and GPU. No explicit cudaMemcpy.
 *      The driver automatically migrates pages to whichever processor
 *      (CPU or GPU) accesses them next -- on-demand page faulting.
 *
 *   2. cudaStreamAttachMemAsync
 *      Controls which CUDA stream "owns" a managed memory allocation.
 *      Two strategies:
 *        cudaMemAttachSingle -- memory is attached to ONE specific stream.
 *                               Other streams/host cannot access it until
 *                               the owning stream finishes. Best for GPU-only
 *                               operations with predictable access pattern.
 *        cudaMemAttachHost  -- memory is accessible from the HOST while
 *                               the GPU is busy with other streams. Allows
 *                               CPU and GPU to concurrently access DIFFERENT
 *                               managed allocations.
 *
 *   3. Task-based dispatching (host vs device)
 *      Small tasks (< 100 elements): run on CPU (latency-bound, GPU overhead
 *      kills benefit). Uses cudaMemAttachHost so CPU can work while GPU
 *      processes other tasks concurrently.
 *      Large tasks (>= 100 elements): run on GPU via cuBLAS. Uses
 *      cudaMemAttachSingle for efficient GPU-side page mapping.
 *
 *   4. OpenMP + multiple CUDA streams
 *      OpenMP creates CPU threads; each thread drives its own CUDA stream
 *      + cuBLAS handle. Tasks are distributed dynamically across threads.
 *
 * Build (Windows MSVC):
 *   nvcc -Xcompiler /openmp -lcublas -o UnifiedMemoryStreams_clean UnifiedMemoryStreams_clean.cu
 */

#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "[CUBLAS ERROR] %s:%d  status=%d\n",              \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ===========================================================================
// Task: holds a square matrix, vector, and result array in Unified Memory
// ===========================================================================
//   Uses cudaMallocManaged -- the same pointer is valid on both CPU and GPU.
//   Page faults on first access trigger automatic migration to the accessing
//   processor. Subsequent accesses to the migrated page are fast (local).
// ===========================================================================
struct Task
{
    int    id;
    int    size;
    double *data;      // size x size matrix (row-major)
    double *result;    // size-element output vector
    double *vector;    // size-element input vector

    Task() : id(0), size(0), data(nullptr), result(nullptr), vector(nullptr) {}

    Task(int sz) : id(0), size(sz), data(nullptr), result(nullptr), vector(nullptr)
    {
        if (sz <= 0) return;

        // Unified Memory: one allocation works on CPU AND GPU
        CUDA_CHECK(cudaMallocManaged(&data,   sizeof(double) * sz * sz));
        CUDA_CHECK(cudaMallocManaged(&result, sizeof(double) * sz));
        CUDA_CHECK(cudaMallocManaged(&vector, sizeof(double) * sz));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~Task()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        if (data)   cudaFree(data);
        if (result) cudaFree(result);
        if (vector) cudaFree(vector);
    }

    // Disable copy (simplifies ownership)
    Task(const Task &) = delete;
    Task &operator=(const Task &) = delete;

    // Move semantics
    Task(Task &&other) noexcept
        : id(other.id), size(other.size),
          data(other.data), result(other.result), vector(other.vector)
    {
        other.data   = nullptr;
        other.result = nullptr;
        other.vector = nullptr;
    }

    // Allocate + initialise with random data
    void init(int sz, int unique_id, std::mt19937 &rng)
    {
        id   = unique_id;
        size = sz;

        CUDA_CHECK(cudaMallocManaged(&data,   sizeof(double) * sz * sz));
        CUDA_CHECK(cudaMallocManaged(&result, sizeof(double) * sz));
        CUDA_CHECK(cudaMallocManaged(&vector, sizeof(double) * sz));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Populate with random data
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < sz * sz; i++)
            data[i] = dist(rng);

        for (int i = 0; i < sz; i++) {
            result[i] = 0.0;
            vector[i] = dist(rng);
        }
    }
};

// ===========================================================================
// Host-side DGEMM-like matrix-vector multiply (row-major)
// ===========================================================================
//   y = alpha * A * x + beta * y
//   Used for small tasks where GPU launch overhead dominates.
// ===========================================================================
static void hostGEMV(int m, int n, double alpha, const double *A,
                      const double *x, double beta, double *y)
{
    for (int i = 0; i < m; i++) {
        y[i] *= beta;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// ===========================================================================
// Execute a task: host (small) or device (large) based on size
// ===========================================================================
//   KEY CONCEPT -- cudaStreamAttachMemAsync:
//     Unified Memory needs to know "who is accessing this data" so the
//     driver can migrate pages efficiently. This function tells CUDA
//     which stream's operations will access the managed memory.
//
//   cudaMemAttachHost:
//     "Host will access this. GPU may NOT use it on any stream."
//     Enables concurrent CPU access while GPU runs on OTHER allocations.
//
//   cudaMemAttachSingle:
//     "Only THIS stream will access it. No one else (not even host)."
//     Best for exclusive GPU use -- the driver optimizes page placement.
//
//   Why we can't just use managed memory without attach?
//     Without explicit attach, the driver uses on-demand page faulting.
//     But when CPU and GPU access DIFFERENT managed allocations
//     simultaneously, the driver needs hints to avoid thrashing
//     (pages bouncing between CPU and GPU on every access).
// ===========================================================================
static void executeTask(Task &t, cublasHandle_t *handles,
                         cudaStream_t *streams, int tid)
{
    if (t.size < 100) {
        // ---- HOST execution ----
        printf("  Task [%2d]  thread %d -> HOST  (size=%d)\n", t.id, tid, t.size);

        // Attach memory to host via a dummy stream (stream[0]).
        // This tells the driver: "CPU will touch these pages now."
        // The host can access t.data/t.vector/t.result while GPU
        // concurrently runs other tasks in other streams.
        CUDA_CHECK(cudaStreamAttachMemAsync(streams[0], t.data,   0,
                                             cudaMemAttachHost));
        CUDA_CHECK(cudaStreamAttachMemAsync(streams[0], t.vector, 0,
                                             cudaMemAttachHost));
        CUDA_CHECK(cudaStreamAttachMemAsync(streams[0], t.result, 0,
                                             cudaMemAttachHost));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // Now do the matrix-vector multiply on the CPU
        hostGEMV(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
    } else {
        // ---- DEVICE execution via cuBLAS ----
        printf("  Task [%2d]  thread %d -> GPU   (size=%d)\n", t.id, tid, t.size);

        // Attach memory exclusively to THIS thread's stream.
        // The driver will migrate pages to GPU memory and optimise
        // for GPU-side access only.
        CUDA_CHECK(cudaStreamAttachMemAsync(streams[tid + 1], t.data,   0,
                                             cudaMemAttachSingle));
        CUDA_CHECK(cudaStreamAttachMemAsync(streams[tid + 1], t.vector, 0,
                                             cudaMemAttachSingle));
        CUDA_CHECK(cudaStreamAttachMemAsync(streams[tid + 1], t.result, 0,
                                             cudaMemAttachSingle));

        // Bind cuBLAS to this thread's stream
        CUBLAS_CHECK(cublasSetStream(handles[tid + 1], streams[tid + 1]));

        // cuBLAS DGEMM-V: result = 1.0 * A * x + 0.0 * result
        //
        // CRITICAL: data is stored ROW-MAJOR (C/C++ convention), but cuBLAS
        // reads in COLUMN-MAJOR (FORTRAN convention).  CUBLAS_OP_T tells
        // cuBLAS to swap the traversal order (row<->col) rather than physically
        // transposing the data.  For a square matrix (size x size), lda=size
        // stays correct in both conventions -- the index arithmetic
        //   row-major A[i*size+j]  ==  cuBLAS_T access A[j + i*size]
        // cancels out perfectly.
        double one  = 1.0;
        double zero = 0.0;
        CUBLAS_CHECK(cublasDgemv(handles[tid + 1], CUBLAS_OP_T,
                                  t.size, t.size,
                                  &one,  t.data,   t.size,
                                         t.vector, 1,
                                  &zero, t.result, 1));
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[UnifiedMemoryStreams] - Starting...\n");

    // ------------------------------------------------------------------
    // 1. Pick device 0, verify Unified Memory support (CC >= 3.0)
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

    if (!prop.managedMemory) {
        printf("ERROR: Unified Memory not supported on this device\n");
        return EXIT_SUCCESS;
    }

    int computeMode = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&computeMode,
                                       cudaDevAttrComputeMode, 0));
    if (computeMode == cudaComputeModeProhibited) {
        printf("ERROR: device in compute-prohibited mode\n");
        return EXIT_SUCCESS;
    }

    // ------------------------------------------------------------------
    // 2. Seed RNG, set up threads/streams/handles
    // ------------------------------------------------------------------
    std::mt19937 rng((unsigned int)
        std::chrono::steady_clock::now().time_since_epoch().count());

    const int nthreads = 4;

    // nthreads+1: index 0 = dummy stream for host attach,
    //             indices 1..nthreads = per-thread GPU streams
    cudaStream_t   *streams = new cudaStream_t[nthreads + 1];
    cublasHandle_t *handles = new cublasHandle_t[nthreads + 1];

    for (int i = 0; i < nthreads + 1; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUBLAS_CHECK(cublasCreate(&handles[i]));
    }

    // ------------------------------------------------------------------
    // 3. Create tasks with random sizes
    // ------------------------------------------------------------------
    const int N = 40;
    Task *taskList = new Task[N];

    printf("\nTask list (%d tasks):\n", N);
    std::uniform_int_distribution<int> sizeDist(64, 1000);
    for (int i = 0; i < N; i++) {
        int sz = sizeDist(rng);
        taskList[i].init(sz, i, rng);
        printf("  Task %2d: size=%4d -> runs on %s\n",
               i, sz, (sz < 100 ? "HOST" : "GPU"));
    }

    // ------------------------------------------------------------------
    // 4. Execute tasks via OpenMP + multiple streams
    // ------------------------------------------------------------------
    //   Each OpenMP thread:
    //     - Sets its own CUDA device (cudaSetDevice)
    //     - Drives its own cuBLAS handle + CUDA stream pair
    //     - Dynamically grabs tasks from the shared task list
    //
    //   schedule(dynamic): tasks are NOT pre-assigned. Threads grab the
    //     next available task as they finish their current one -- load
    //     balancing without any extra code.
    // ------------------------------------------------------------------
    printf("\nExecuting tasks on host / device...\n\n");

    omp_set_num_threads(nthreads);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        CUDA_CHECK(cudaSetDevice(0));
        int tid = omp_get_thread_num();
        executeTask(taskList[i], handles, streams, tid);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // 5. Cleanup
    // ------------------------------------------------------------------
    delete[] taskList;

    for (int i = 0; i < nthreads + 1; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUBLAS_CHECK(cublasDestroy(handles[i]));
    }
    delete[] streams;
    delete[] handles;

    printf("\nAll Done!\n");
    return EXIT_SUCCESS;
}
