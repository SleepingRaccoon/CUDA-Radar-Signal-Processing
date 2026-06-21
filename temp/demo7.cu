/*
nvcc -o demo7 demo7.cu -lcublas 
.\demo7
*/


#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

// ============== WMMA params (adjust per GPU arch) ==============
// Volta (SM70):    16x16x16
// Turing (SM75):   16x16x16
// Ampere (SM80/86): 16x16x16, 16x8x16, 16x8x8, 8x32x16
// Hopper (SM90):    same as above + more variants
// Note: these sizes are hardware-fixed, cannot be arbitrarily set
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;
constexpr int M = 1024;
constexpr int N = 512;
constexpr int K = 256;
constexpr int RUNS = 10;

// ============== utility functions ==============
void initMatrix(float* mat, int rows, int cols, float scale = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dis(gen) * scale;
    }
}

void printHeader() {
    std::cout << "\n";
    std::cout << "=============================================================================================\n";
    std::cout << "                          GPU GEMM Benchmark Report (Matrix " << M << "x" << N << " x " << K << ")\n";
    std::cout << "=============================================================================================\n\n";
}

void printResults(const std::string& name, double time_ms, double flops, double throughput_gb, double peak_gflops) {
    std::cout << std::left << std::setw(25) << name
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << time_ms << " ms"
              << std::setw(18) << std::fixed << std::setprecision(2) << flops << " GFLOP/s"
              << std::setw(18) << std::fixed << std::setprecision(2) << throughput_gb << " GB/s"
              << std::setw(18) << std::fixed << std::setprecision(2) << peak_gflops << " GFLOP/s"
              << "\n";
}

void printSeparator() {
    std::cout << "---------------------------------------------------------------------------------------------\n";
}

void printTableHeader() {
    std::cout << std::left << std::setw(25) << "Method"
              << std::right << std::setw(20) << "Time (ms)"
              << std::setw(20) << "GFLOPS"
              << std::setw(20) << "Bandwidth (GB/s)"
              << std::setw(20) << "Peak GFLOPS"
              << "\n";
    printSeparator();
}

// ============== CPU GEMM ==============
void cpuGemm(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

double benchmarkCpuGemm(const float* A, const float* B, float* C, int m, int n, int k) {
    auto start = std::chrono::high_resolution_clock::now();
    cpuGemm(A, B, C, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============== cuBLAS GEMM ==============
double benchmarkCublasGemm(const float* A, const float* B, float* C, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < RUNS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= RUNS;
    
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time_ms;
}

// ============== WMMA GEMM Kernel ==============
__global__ void wmmaGemmKernel(half* A, half* B, float* C, int m, int n, int k) {
    // WMMA fragment sizes controlled by compile-time constexpr
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, nvcuda::wmma::row_major> A_frag;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, nvcuda::wmma::col_major> B_frag;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, float> C_frag;

    int block_m = blockIdx.y * MMA_M;
    int block_n = blockIdx.x * MMA_N;

    if (block_m >= m || block_n >= n) return;

    A_frag a_frag;
    B_frag b_frag;
    C_frag c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k_start = 0; k_start < k; k_start += MMA_K) {
        nvcuda::wmma::load_matrix_sync(a_frag, &A[block_m * k + k_start], k);
        nvcuda::wmma::load_matrix_sync(b_frag, &B[k_start * n + block_n], n);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    nvcuda::wmma::store_matrix_sync(&C[block_m * n + block_n], c_frag, n, nvcuda::wmma::mem_row_major);
}

// ============== wrapper ==============
double benchmarkWmmaGemm(const half* A, const half* B, float* C, int m, int n, int k) {
    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, m * k * sizeof(half));
    cudaMalloc(&d_B, k * n * sizeof(half));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    cudaMemcpy(d_A, A, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 blocks((n + MMA_N - 1) / MMA_N, (m + MMA_M - 1) / MMA_M);
    dim3 threads(32, 1);
    
    // Warmup
    wmmaGemmKernel<<<blocks, threads>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "WMMA Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < RUNS; ++i) {
        wmmaGemmKernel<<<blocks, threads>>>(d_A, d_B, d_C, m, n, k);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= RUNS;
    
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time_ms;
}

// ============== metrics calculation ==============
void calculateMetrics(double time_ms, int m, int n, int k, 
                      double& flops, double& throughput_gb, double& peak_gflops) {
    double flops_count = 2.0 * m * n * k;
    flops = flops_count / (time_ms / 1000.0) / 1e9;
    
    double bytes = (m * k + k * n + m * n) * sizeof(float);
    throughput_gb = bytes / (time_ms / 1000.0) / 1e9;
    
    peak_gflops = flops;
}

// ============== verification (relative error) ==============
// FP32 ref vs FP32 (cuBLAS): eps = 1e-5
// FP32 ref vs FP16 (WMMA):  eps = 1e-2 (FP16 precision ~3-4 decimal digits)
bool verifyResults(const float* C1, const float* C2, int size, float eps = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(C1[i] - C2[i]);
        float max_val = fmaxf(fabsf(C1[i]), fabsf(C2[i]));
        bool ok = (max_val > 1.0f) ? (diff / max_val <= eps) : (diff <= eps);
        if (!ok) {
            std::cout << "Mismatch at index " << i << ": " << C1[i] << " vs " << C2[i]
                      << ", rel_err=" << (max_val > 1.0f ? diff / max_val : diff) << std::endl;
            return false;
        }
    }
    return true;
}

// ============== GPU info (fixed) ==============
void printGpuInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n=============================================================================================\n";
    std::cout << "                                    GPU Information\n";
    std::cout << "=============================================================================================\n\n";
    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    
    // CUDA 12+ removed prop.clockRate / prop.memoryClockRate
    // use cudaDeviceGetAttribute instead (supports all CUDA versions)
    int clock_rate_khz = 0, mem_clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clock_rate_khz, cudaDevAttrMemoryClockRate, 0);

    std::cout << "Max Clock Rate: " << clock_rate_khz / 1000 << " MHz\n";
    std::cout << "Memory Clock Rate: " << mem_clock_rate_khz / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
    
    // theoretical peak (FP32 FMA: 2 ops/cycle * 128 FP32 cores/SM * SMs * freq)
    // note: FP32 cores per SM varies by arch (Turing: 64, Ampere: 128, Hopper: 128)
    // this is a rough estimate
    double clock_ghz = static_cast<double>(clock_rate_khz) / 1e6;  // kHz -> GHz
    double theoretical_peak = prop.multiProcessorCount * 128 * 2 * clock_ghz / 1000.0;  // TFLOPS
    std::cout << "Theoretical Peak FP32 (est): ~" << std::fixed << std::setprecision(1) 
              << theoretical_peak << " TFLOPS\n";
}

// ============== main ==============
int main() {
    std::cout << "Initializing matrices...\n";
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_cublas(M * N);
    std::vector<float> h_C_wmma(M * N);
    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    
    initMatrix(h_A.data(), M, K);
    initMatrix(h_B.data(), K, N);
    
    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B[i]);
    
    printHeader();
    printTableHeader();
    
    // ====== CPU GEMM ======
    std::cout << "Running CPU GEMM... (this may take a while)\n";
    double cpu_time = 0.0;
    for (int i = 0; i < RUNS; ++i) {
        cpu_time += benchmarkCpuGemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    }
    cpu_time /= RUNS;
    
    double cpu_flops, cpu_bw, cpu_peak;
    calculateMetrics(cpu_time, M, N, K, cpu_flops, cpu_bw, cpu_peak);
    printResults("CPU GEMM", cpu_time, cpu_flops, cpu_bw, cpu_peak);
    
    // ====== cuBLAS GEMM ======
    std::cout << "Running cuBLAS GEMM...\n";
    double cublas_time = benchmarkCublasGemm(h_A.data(), h_B.data(), h_C_cublas.data(), M, N, K);
    
    double cublas_flops, cublas_bw, cublas_peak;
    calculateMetrics(cublas_time, M, N, K, cublas_flops, cublas_bw, cublas_peak);
    printResults("cuBLAS GEMM", cublas_time, cublas_flops, cublas_bw, cublas_peak);
    
    // ====== WMMA GEMM ======
    std::cout << "Running WMMA GEMM...\n";
    double wmma_time = benchmarkWmmaGemm(h_A_half.data(), h_B_half.data(), h_C_wmma.data(), M, N, K);
    
    double wmma_flops, wmma_bw, wmma_peak;
    calculateMetrics(wmma_time, M, N, K, wmma_flops, wmma_bw, wmma_peak);
    printResults("WMMA GEMM", wmma_time, wmma_flops, wmma_bw, wmma_peak);
    
    printSeparator();
    
    // ====== verification ======
    std::cout << "\nVerifying results...\n";
    bool cpu_vs_cublas = verifyResults(h_C_cpu.data(), h_C_cublas.data(), M * N);
    bool cpu_vs_wmma = verifyResults(h_C_cpu.data(), h_C_wmma.data(), M * N, 1e-2);
    
    std::cout << "CPU vs cuBLAS: " << (cpu_vs_cublas ? "PASS [OK]" : "FAIL [X]") << "\n";
    std::cout << "CPU vs WMMA:   " << (cpu_vs_wmma ? "PASS [OK]" : "FAIL [X]") << "\n";
    
    // ====== speedup ======
    std::cout << "\n";
    std::cout << "=============================================================================================\n";
    std::cout << "                                    Speedup Analysis\n";
    std::cout << "=============================================================================================\n\n";
    std::cout << std::left << std::setw(25) << "Comparison"
              << std::right << std::setw(20) << "Speedup"
              << "\n";
    printSeparator();
    std::cout << std::left << std::setw(25) << "cuBLAS vs CPU"
              << std::right << std::setw(20) << std::fixed << std::setprecision(2) 
              << (cpu_time / cublas_time) << "x"
              << "\n";
    std::cout << std::left << std::setw(25) << "WMMA vs CPU"
              << std::right << std::setw(20) << std::fixed << std::setprecision(2) 
              << (cpu_time / wmma_time) << "x"
              << "\n";
    std::cout << std::left << std::setw(25) << "WMMA vs cuBLAS"
              << std::right << std::setw(20) << std::fixed << std::setprecision(2) 
              << (cublas_time / wmma_time) << "x"
              << "\n";
    printSeparator();
    
    // ====== GPU info ======
    printGpuInfo();
    
    std::cout << "\n=============================================================================================\n";
    std::cout << "                                      Summary\n";
    std::cout << "=============================================================================================\n\n";
    std::cout << "Matrix Size: " << M << "x" << N << " x " << K << "\n";
    std::cout << "Total Operations: " << std::fixed << std::setprecision(2) 
              << (2.0 * M * N * K / 1e9) << " GFLOPS\n";
    std::cout << "Total Memory Traffic: " << std::fixed << std::setprecision(3) 
              << ((M*K + K*N + M*N) * sizeof(float) / 1e9) << " GB\n";
    std::cout << "Runs averaged: " << RUNS << "\n";
    
    std::cout << "\n=============================================================================================\n";
    
    return 0;
}