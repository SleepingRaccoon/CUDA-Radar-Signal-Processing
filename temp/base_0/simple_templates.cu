/*
nvcc -o simple_templates simple_templates.cu
simple_templates.exe
*/

/**
 * simpleTemplates_clean.cu
 *
 * Single-file version of NVIDIA's simpleTemplates sample.
 *
 * Core concept -- C++ templates in CUDA kernels:
 *   Write a kernel once as a template, then instantiate it for float, int,
 *   double, etc. without duplicating code. This is standard C++ template
 *   programming, but applied to __global__ functions.
 *
 *   KEY PROBLEM solved by SharedMemory<T>:
 *     Dynamic shared memory uses `extern __shared__ T sdata[]`, but you
 *     cannot directly declare `extern __shared__` in a template -- each
 *     template instantiation would produce a conflicting external symbol
 *     with the same type-dependent name. The compiler/linker needs
 *     concrete, differently-named symbols per type.
 *
 *     Solution: SharedMemory<T> uses explicit specialization -- each type
 *     (int, float, double, ...) gets its own struct with a differently-
 *     named extern shared array (s_int[], s_float[], s_double[], ...).
 *     The compiler then sees unique symbols and compiles cleanly.
 *
 *   This sample demonstrates:
 *     1. Templated __global__ kernel with dynamic shared memory
 *     2. SharedMemory<T> pattern for type-safe shared memory access
 *     3. Calling the same kernel with different types (float, int)
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

// ===========================================================================
// SharedMemory<T> -- dynamic shared memory for templated kernels
// ===========================================================================
//
//  Because `extern __shared__ T sdata[]` in a template kernel produces
//  duplicate-symbol errors across instantiations, we wrap it in a struct
//  with explicit specializations. Each specialization uses a unique symbol
//  name (s_int, s_float, ...), avoiding linker conflicts.
//
//  Usage inside a templated kernel:
//      SharedMemory<T> smem;
//      T *sdata = smem.getPointer();   // replaces extern __shared__ T sdata[]
//
//  The base (unspecialized) template has an intentional compile error
//  to prevent instantiation with unsupported types.
// ===========================================================================
template <typename T>
struct SharedMemory
{
    // Deliberately broken -- forces user to specialize for each type
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return nullptr;
    }
};

// Explicit specializations for all common types

template <> struct SharedMemory<int>
{
    __device__ int *getPointer() {
        extern __shared__ int s_int[];
        return s_int;
    }
};

template <> struct SharedMemory<unsigned int>
{
    __device__ unsigned int *getPointer() {
        extern __shared__ unsigned int s_uint[];
        return s_uint;
    }
};

template <> struct SharedMemory<char>
{
    __device__ char *getPointer() {
        extern __shared__ char s_char[];
        return s_char;
    }
};

template <> struct SharedMemory<unsigned char>
{
    __device__ unsigned char *getPointer() {
        extern __shared__ unsigned char s_uchar[];
        return s_uchar;
    }
};

template <> struct SharedMemory<short>
{
    __device__ short *getPointer() {
        extern __shared__ short s_short[];
        return s_short;
    }
};

template <> struct SharedMemory<unsigned short>
{
    __device__ unsigned short *getPointer() {
        extern __shared__ unsigned short s_ushort[];
        return s_ushort;
    }
};

template <> struct SharedMemory<long>
{
    __device__ long *getPointer() {
        extern __shared__ long s_long[];
        return s_long;
    }
};

template <> struct SharedMemory<unsigned long>
{
    __device__ unsigned long *getPointer() {
        extern __shared__ unsigned long s_ulong[];
        return s_ulong;
    }
};

template <> struct SharedMemory<bool>
{
    __device__ bool *getPointer() {
        extern __shared__ bool s_bool[];
        return s_bool;
    }
};

template <> struct SharedMemory<float>
{
    __device__ float *getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <> struct SharedMemory<double>
{
    __device__ double *getPointer() {
        extern __shared__ double s_double[];
        return s_double;
    }
};

// ===========================================================================
// Templated kernel
// ===========================================================================
//   Shared memory usage pattern:
//     (1) Each thread loads one element from global memory into shared
//     (2) __syncthreads() -- ensure all loads complete
//     (3) Each thread multiplies its element by num_threads
//     (4) __syncthreads() -- ensure all computation completes
//     (5) Each thread stores result back to global memory
//
//   The third launch argument (smem_bytes) = num_threads * sizeof(T).
// ===========================================================================
template <class T>
__global__ void testKernel(const T *g_idata, T *g_odata, int num_threads)
{
    // Replace: extern __shared__ T sdata[];
    SharedMemory<T> smem;
    T *sdata = smem.getPointer();

    int tid = threadIdx.x;

    // Load global -> shared
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // Compute on shared memory
    sdata[tid] = (T)num_threads * sdata[tid];
    __syncthreads();

    // Store shared -> global
    g_odata[tid] = sdata[tid];
}

// ===========================================================================
// CPU reference: each element * len
// ===========================================================================
template <class T>
static void computeGold(T *reference, const T *idata, int len)
{
    T T_len = (T)len;
    for (int i = 0; i < len; ++i) {
        reference[i] = idata[i] * T_len;
    }
}

// ===========================================================================
// Templated test runner: instantiated separately for each type T
// ===========================================================================
template <class T>
static int runTest(int len)
{
    int mem_size = (int)(sizeof(T) * len);

    dim3 grid(1);
    dim3 block(len);
    int smem_bytes = mem_size;   // one element per thread in shared memory

    printf("  Type=%s, num_threads=%d, smem=%d bytes\n",
           sizeof(T) == 4 ? "float/int" : "double", len, smem_bytes);

    // ------------------------------------------------------------------
    // Allocate + initialize host memory
    // ------------------------------------------------------------------
    T *h_idata = new T[len];
    for (int i = 0; i < len; ++i) {
        h_idata[i] = (T)i;
    }

    // ------------------------------------------------------------------
    // Allocate device memory + copy input
    // ------------------------------------------------------------------
    T *d_idata = nullptr;
    T *d_odata = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_idata, mem_size));
    CUDA_CHECK(cudaMalloc((void **)&d_odata, mem_size));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Warm-up + timed launch
    // ------------------------------------------------------------------
    testKernel<T><<<grid, block, smem_bytes>>>(d_idata, d_odata, len);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());   // catch kernel launch errors

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    testKernel<T><<<grid, block, smem_bytes>>>(d_idata, d_odata, len);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // ------------------------------------------------------------------
    // Copy result back + CPU reference
    // ------------------------------------------------------------------
    T *h_odata = new T[len];
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    T *reference = new T[len];
    computeGold<T>(reference, h_idata, len);

    // ------------------------------------------------------------------
    // Verify
    // ------------------------------------------------------------------
    bool passed = true;
    for (int i = 0; i < len; ++i) {
        // Allow small tolerance for float, exact match for int
        double diff = (double)reference[i] - (double)h_odata[i];
        if (diff < 0.0) diff = -diff;
        double tol = (sizeof(T) == 8) ? 1e-10 : 1e-4;
        if (diff > tol) {
            printf("    FAIL at [%d]: CPU=%.4f, GPU=%.4f\n",
                   i, (double)reference[i], (double)h_odata[i]);
            passed = false;
            break;
        }
    }
    printf("    Time:   %.4f ms\n", kernel_ms);
    printf("    Result: %s\n", passed ? "PASS" : "FAIL");

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    delete[] h_idata;
    delete[] h_odata;
    delete[] reference;
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return passed ? 0 : 1;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("[simpleTemplates] - Starting...\n\n");

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
    printf("Device %d: \"%s\", SM %d.%d, %d MPs\n\n",
           0, props.name, props.major, props.minor,
           props.multiProcessorCount);

    // ------------------------------------------------------------------
    // 2. Run tests with different types
    // ------------------------------------------------------------------
    //   The same testKernel template is instantiated for TWO different
    //   types. Each instantiation gets its own compiled kernel binary.
    //   This demonstrates code reuse via templates in CUDA.
    // ------------------------------------------------------------------
    int failures = 0;

    printf("> runTest<float, 32>\n");
    failures += runTest<float>(32);

    printf("\n> runTest<int, 64>\n");
    failures += runTest<int>(64);

    printf("\n[simpleTemplates] -> Test Results: %d Failures\n", failures);

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
