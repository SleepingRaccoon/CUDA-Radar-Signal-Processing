#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CHECK_CUDA(call) {                                                    \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                     \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

// CPU version
void sgemv_cpu(float *A, float *x, float *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * x[j];
        y[i] = sum;
    }
}

// GPU version
__global__ void sgemv_v1(float *A, float *x, float *y, int M, int N) {
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int row = blockIdx.x * warps_per_block + warp_id;
    
    if (row >= M)
        return;
    
    float res = 0.0f;
    int it_num = CEIL_DIV(N, warpSize);

    for (int i = 0; i < it_num; i++) {
        int col = i * warpSize + lane_id;
        if (col < N)
            res += A[row * N + col] * x[col];
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }

    if (lane_id == 0)
        y[row] = res;
}

int main() {
    // Matrix dimensions (non-power-of-two)
    int M = 7193;
    int N = 8147;
    
    size_t size_A = M * N * sizeof(float);
    size_t size_x = N * sizeof(float);
    size_t size_y = M * sizeof(float);
    
    // Host memory allocation
    float *h_A = (float*)malloc(size_A);
    float *h_x = (float*)malloc(size_x);
    float *h_y_cpu = (float*)malloc(size_y);
    float *h_y_gpu = (float*)malloc(size_y);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * N; i++) h_A[i] = dist(gen);
    for (int i = 0; i < N; i++) h_x[i] = dist(gen);
    
    // Device memory allocation
    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_x, size_x));
    CHECK_CUDA(cudaMalloc(&d_y, size_y));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    
    // ==================== CPU Computation ====================
    auto cpu_start = std::chrono::high_resolution_clock::now();
    sgemv_cpu(h_A, h_x, h_y_cpu, M, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // ==================== GPU Computation ====================
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = CEIL_DIV(M, warps_per_block);
    
    dim3 grid(blocks);
    dim3 block(threads_per_block);
    
    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // // Warmup
    // sgemv_v1<<<grid, block>>>(d_A, d_x, d_y, M, N);
    // CHECK_CUDA(cudaDeviceSynchronize());
    
    // Timed run
    CHECK_CUDA(cudaEventRecord(start, 0));
    sgemv_v1<<<grid, block>>>(d_A, d_x, d_y, M, N);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float gpu_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, size_y, cudaMemcpyDeviceToHost));
    
    // ==================== Verify Correctness ====================
    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        float error = fabs(h_y_cpu[i] - h_y_gpu[i]);
        if (error > max_error) max_error = error;
    }
    
    // ==================== Output Results ====================
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========== SGEMV Performance ==========" << std::endl;
    std::cout << "Matrix size: " << M << " x " << N << std::endl;
    std::cout << "Threads per block: " << threads_per_block << std::endl;
    std::cout << "Blocks: " << blocks << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup:  " << cpu_time / gpu_time_ms << "x" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Max Error (CPU vs GPU): " << max_error << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_A);
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    
    return 0;
}