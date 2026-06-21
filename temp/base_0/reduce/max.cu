#include <cstdio>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

float vec_max_cpu(float *a, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (a[i] > max_val) 
            max_val = a[i];
    }
    return max_val;
}

__device__ float atomic_max(float *ptr, float val) {
    int *ptr_as_int = reinterpret_cast<int*>(ptr);
    int old = *ptr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(ptr_as_int, assumed, 
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void vec_max_v1(float *a, int n, float *ptr) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;

    __shared__ float sh_mem[32];
    
    float val = (idx < n) ? a[idx] : -FLT_MAX;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

    if (lane_id == 0)
        sh_mem[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < warps_per_block) ? sh_mem[lane_id] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) 
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0)
            atomic_max(ptr, val);
    }
}

int main() {
    // Random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    
    // Random array size between 1M and 32M
    std::uniform_int_distribution<int> size_dist(5000'0000, 6000'0000);
    int N = size_dist(rng);
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_a = new float[N];
    float* h_result = new float;

    // Initialize with random values
    for (int i = 0; i < N; i++) {
        h_a[i] = dist(rng);
    }
    // Insert a known maximum for verification
    h_a[N / 2] = 9999.0f;

    // CPU version timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = vec_max_cpu(h_a, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // GPU memory allocation
    float *d_a = nullptr;
    float *d_result = nullptr;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    float init_val = -FLT_MAX;
    cudaMemcpy(d_result, &init_val, sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = (threads_per_block / 32) * sizeof(float);

    // GPU timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vec_max_v1<<<blocks, threads_per_block, shared_mem_size>>>(d_a, N, d_result);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_result = *h_result;

    // Error calculation
    float absolute_error = std::fabs(cpu_result - gpu_result);
    float relative_error = absolute_error / std::fabs(cpu_result);

    // Output
    std::printf("Array size: %d elements (%.2f MB)\n", N, static_cast<double>(size) / (1024 * 1024));
    std::printf("CPU result: %.6f, time: %.3f ms\n", cpu_result, cpu_time);
    std::printf("GPU result: %.6f, time: %.3f ms\n", gpu_result, gpu_time);
    std::printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    std::printf("Absolute error: %e\n", absolute_error);
    std::printf("Relative error: %e\n", relative_error);

    // Cleanup
    delete[] h_a;
    delete h_result;
    cudaFree(d_a);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}