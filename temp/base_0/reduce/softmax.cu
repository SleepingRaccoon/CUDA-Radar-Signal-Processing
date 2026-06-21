#include <cstdio>
#include <random>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

// ========== CPU reference ==========
void softmax_cpu(float *x, float *y, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++)
        m = fmaxf(m, x[i]);

    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += expf(x[i] - m);
    
    for (int i = 0; i < n; i++)
        y[i] = expf(x[i] - m) / s;
}

// ========== GPU device functions ==========
__device__ float atomic_max(float *ptr, float val) {
    int *ptr_as_int = reinterpret_cast<int*>(ptr);
    int old = *ptr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(ptr_as_int, assumed, 
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while(assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float *x, int n, float *max_val) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;

    __shared__ float sh_mem[32];

    float val = (idx < n) ? x[idx] : -FLT_MAX;
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
            atomic_max(max_val, val);
    }
}

__global__ void sum_kernel(float *x, int n, float *max_val, float *sum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;

    __shared__ float sh_mem[32];

    float val = (idx < n) ? expf(x[idx] - *max_val) : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
   
    if (lane_id == 0)
        sh_mem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < warps_per_block) ? sh_mem[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            atomicAdd(sum, val);
    }
}

__global__ void softmax_kernel(float *x, float *y, int n, float *max_val, float *sum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    y[idx] = expf(x[idx] - *max_val) / *sum;
}

// ========== Main ==========
int main() {
    // Random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_int_distribution<int> size_dist(10000000, 20000000);
    int N = size_dist(rng);
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_x = new float[N];
    float *h_y_cpu = new float[N];
    float *h_y_gpu = new float[N];

    // Initialize with random values
    for (int i = 0; i < N; i++) {
        h_x[i] = dist(rng);
    }

    // ===== CPU =====
    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_cpu(h_x, h_y_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ===== GPU =====
    float *d_x, *d_y, *d_max_val, *d_sum;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = (threads_per_block / 32) * sizeof(float);

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cudaEventRecord(gpu_start);

    // Copy input data to GPU
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Step 1: find max, result stays in d_max_val
    float init_max = -FLT_MAX;
    cudaMemcpy(d_max_val, &init_max, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_x, N, d_max_val);

    // Step 2: compute sum, reads d_max_val directly from device memory
    float init_sum = 0.0f;
    cudaMemcpy(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice);
    sum_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_x, N, d_max_val, d_sum);

    // Step 3: softmax, reads both d_max_val and d_sum from device memory
    softmax_kernel<<<blocks, threads_per_block>>>(d_x, d_y, N, d_max_val, d_sum);

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);

    // Copy result back
    cudaMemcpy(h_y_gpu, d_y, size, cudaMemcpyDeviceToHost);

    // ===== Error analysis =====
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    int max_error_idx = 0;
    float max_error_cpu_val = 0.0f;
    float max_error_gpu_val = 0.0f;

    for (int i = 0; i < N; i++) {
        float abs_err = std::fabs(h_y_cpu[i] - h_y_gpu[i]);
        float rel_err = abs_err / (std::fabs(h_y_cpu[i]) + 1e-10f);
        if (abs_err > max_abs_error) {
            max_abs_error = abs_err;
            max_rel_error = rel_err;
            max_error_idx = i;
            max_error_cpu_val = h_y_cpu[i];
            max_error_gpu_val = h_y_gpu[i];
        }
    }

    // ===== Output =====
    std::printf("Array size: %d elements (%.2f MB)\n", N, static_cast<double>(size) / (1024 * 1024));
    std::printf("CPU time: %.3f ms\n", cpu_time);
    std::printf("GPU time: %.3f ms\n", gpu_time);
    std::printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    std::printf("--- Error Report ---\n");
    std::printf("Max absolute error: %e at index %d\n", max_abs_error, max_error_idx);
    std::printf("  CPU value: %.10f\n", max_error_cpu_val);
    std::printf("  GPU value: %.10f\n", max_error_gpu_val);
    std::printf("Max relative error: %e\n", max_rel_error);

    // Cleanup
    delete[] h_x;
    delete[] h_y_cpu;
    delete[] h_y_gpu;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_max_val);
    cudaFree(d_sum);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    return 0;
}