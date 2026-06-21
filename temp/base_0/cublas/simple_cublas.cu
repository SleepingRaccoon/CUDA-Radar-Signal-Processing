/*
nvcc -o simple_cublas simple_cublas.cu -lcublas
simple_cublas.exe

nvcc -o demo2 demo2.cu -lcublas
demo2.exe
*/

/**
 *
 * Single-file version of NVIDIA's simpleCUBLAS sample.
 *
 * Core concept -- cuBLAS SGEMM (single-precision matrix multiply):
 *   C(M×N) = alpha * A(M×K) * B(K×N) + beta * C(M×N)
 *
 *   cuBLAS is NVIDIA's BLAS library.  The key difference from writing
 *   your own kernel: cuBLAS kernels are hand-tuned per GPU architecture
 *   and almost always faster than a naive custom SGEMM.
 *
 *   CRITICAL convention -- COLUMN-MAJOR (FORTRAN) layout:
 *     cuBLAS inherits BLAS's column-major convention.  A_{ij} is at
 *     A[j * lda + i].  This sample stores data in column-major format
 *     and uses CUBLAS_OP_N (no transpose) for both operands.
 *
 *     If your data is row-major (C/C++ convention), use CUBLAS_OP_T
 *     instead to swap traversal order without physically transposing.
 *
 *   This example uses distinct M, N, K to show the general GEMM signature.
 *
 *   GEMM FLOP count: M*N result elements, each needs K multiply-adds
 *     → 2 * M * N * K FLOPs  (1 multiply + 1 add per inner product)
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__              \
                    << "  " << cudaGetErrorString(err) << std::endl;        \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

#define CUBLAS_CHECK(call) do {                                             \
    cublasStatus_t st = (call);                                             \
    if (st != CUBLAS_STATUS_SUCCESS) {                                      \
        std::cerr << "[CUBLAS ERROR] " << __FILE__ << ":" << __LINE__       \
                    << "  status=" << static_cast<int>(st) << std::endl;    \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

// ===========================================================================
// CPU reference: column-major SGEMM  (matches cuBLAS CUBLAS_OP_N)
// ===========================================================================
//   C(M×N) = alpha * A(M×K) * B(K×N) + beta * C(M×N)
//
//   Column-major layout:
//     A(row,col) = A[row + col * lda]   lda >= M
//     B(row,col) = B[row + col * ldb]   ldb >= K
//     C(row,col) = C[row + col * ldc]   ldc >= M
//
//   C(i,j) = sum_{k=0}^{K-1} A(i,k) * B(k,j)
// ===========================================================================
static void sgemm_cpu(int M, int N, int K,
                       float alpha, const float *A, const float *B,
                       float beta, float *C)
{
    for (int j = 0; j < N; ++j) {          // column of C and B
        for (int i = 0; i < M; ++i) {      // row of C and A
            float prod = 0.0f;
            for (int k = 0; k < K; ++k) {  // inner dimension
                prod += A[k * M + i] * B[j * K + k];
                //       A(k,i)          B(k,j)
            }
            C[j * M + i] = alpha * prod + beta * C[j * M + i];
        }
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    // Distinct dimensions so the GEMM signature is clearly visible.
    //   A: 128 x 256   (M x K)
    //   B: 256 x 512   (K x N)
    //   C: 128 x 512   (M x N)  ← result
    constexpr int M  = 128;
    constexpr int N  = 512;
    constexpr int K  = 256;

    constexpr int lenA  = M * K;
    constexpr int lenB  = K * N;
    constexpr int lenC  = M * N;

    size_t bytesA = lenA * sizeof(float);
    size_t bytesB = lenB * sizeof(float);
    size_t bytesC = lenC * sizeof(float);

    // ------------------------------------------------------------------
    // 1. Device info
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "========================================" << std::endl;
    std::cout << "  simpleCUBLAS -- cuBLAS SGEMM" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU    : " << prop.name << std::endl;
    std::cout << "  A      : " << M << " x " << K
              << "  (" << bytesA / 1024 << " KB)" << std::endl;
    std::cout << "  B      : " << K << " x " << N
              << "  (" << bytesB / 1024 << " KB)" << std::endl;
    std::cout << "  C      : " << M << " x " << N
              << "  (" << bytesC / 1024 << " KB)" << std::endl;
    std::cout << "  FLOPs  : 2 * M * N * K = "
              << (2.0 * M * N * K) / 1e6 << " MFLOPs" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 2. Create cuBLAS handle
    // ------------------------------------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ------------------------------------------------------------------
    // 3. Allocate + init host memory (column-major)
    // ------------------------------------------------------------------
    auto *h_A     = new float[lenA];
    auto *h_B     = new float[lenB];
    auto *h_C     = new float[lenC];
    auto *h_C_ref = new float[lenC];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < lenA; ++i) h_A[i] = dist(rng);
    for (int i = 0; i < lenB; ++i) h_B[i] = dist(rng);
    for (int i = 0; i < lenC; ++i) h_C[i] = dist(rng);

    // ------------------------------------------------------------------
    // 4. Allocate device memory
    // ------------------------------------------------------------------
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // Upload: cublasSetVector (convenience wrapper over cudaMemcpyAsync)
    CUBLAS_CHECK(cublasSetVector(lenA, sizeof(float), h_A, 1, d_A, 1));
    CUBLAS_CHECK(cublasSetVector(lenB, sizeof(float), h_B, 1, d_B, 1));
    CUBLAS_CHECK(cublasSetVector(lenC, sizeof(float), h_C, 1, d_C, 1));

    // ------------------------------------------------------------------
    // 5. CPU reference (timed)
    // ------------------------------------------------------------------
    for (int i = 0; i < lenC; ++i)
        h_C_ref[i] = h_C[i];

    std::cout << "--- CPU SGEMM ---" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    sgemm_cpu(M, N, K, 1.0f, h_A, h_B, 0.0f, h_C_ref);
    auto t1 = std::chrono::high_resolution_clock::now();

    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gflopsCpu = (2.0 * M * N * K) / (cpuMs * 1e-3) / 1e9;

    std::cout << "  Time  : " << cpuMs << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflopsCpu << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 6. Warm-up GPU SGEMM (first cuBLAS call = JIT compile + init)
    // ------------------------------------------------------------------
    float alpha = 1.0f;
    float beta  = 0.0f;

    // lda/ldb/ldc = leading dimension, must be >= number of rows
    //   lda >= M,  ldb >= K,  ldc >= M
    CUBLAS_CHECK(cublasSgemm(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,   // A, B: no transpose
                              M, N, K,                     // M, N, K
                              &alpha,
                              d_A, M,                       // A: M×K, lda=M
                              d_B, K,                       // B: K×N, ldb=K
                              &beta,
                              d_C, M));                     // C: M×N, ldc=M
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // 7. Timed GPU SGEMM
    // ------------------------------------------------------------------
    std::cout << "--- cuBLAS SGEMM ---" << std::endl;

    // Re-upload initial C (overwritten by warm-up)
    CUBLAS_CHECK(cublasSetVector(lenC, sizeof(float), h_C, 1, d_C, 1));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,
                              &alpha,
                              d_A, M,
                              d_B, K,
                              &beta,
                              d_C, M));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double gflopsGpu = (2.0 * M * N * K) / (gpuMs * 1e-3) / 1e9;

    std::cout << "  Time  : " << gpuMs << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflopsGpu << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 8. Download result + verify
    // ------------------------------------------------------------------
    std::cout << "--- Verification ---" << std::endl;

    auto *h_C_gpu = new float[lenC];
    CUBLAS_CHECK(cublasGetVector(lenC, sizeof(float), d_C, 1, h_C_gpu, 1));

    double errorNorm = 0.0;
    double refNorm   = 0.0;

    for (int i = 0; i < lenC; ++i) {
        double diff = static_cast<double>(h_C_ref[i])
                     - static_cast<double>(h_C_gpu[i]);
        errorNorm += diff * diff;
        refNorm   += static_cast<double>(h_C_ref[i])
                     * static_cast<double>(h_C_ref[i]);
    }

    errorNorm = std::sqrt(errorNorm);
    refNorm   = std::sqrt(refNorm);

    bool pass = (errorNorm / refNorm < 1.0e-5);

    std::cout << "  L2 error : " << std::scientific << errorNorm << std::fixed
              << std::endl;
    std::cout << "  L2 norm  : " << refNorm << std::endl;
    std::cout << "  Rel error: " << std::scientific << errorNorm / refNorm
              << std::fixed << std::endl;
    std::cout << "  Result   : " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 9. Summary
    // ------------------------------------------------------------------
    std::cout << "========================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left
              << std::setw(14) << "Version"
              << std::right
              << std::setw(10) << "Time(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "vs CPU/x"
              << std::endl;

    std::cout << std::left  << std::setw(14) << "CPU"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << cpuMs
              << std::setw(12) << std::setprecision(1) << gflopsCpu
              << std::setw(11) << "1.0 x"
              << std::endl;

    std::cout << std::left  << std::setw(14) << "cuBLAS"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << gpuMs
              << std::setw(12) << std::setprecision(1) << gflopsGpu
              << std::setw(11) << std::setprecision(1)
              << cpuMs / gpuMs << " x"
              << (pass ? "" : "  !!FAIL!!")
              << std::endl;

    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 10. Cleanup
    // ------------------------------------------------------------------
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_C_gpu;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(handle));

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
