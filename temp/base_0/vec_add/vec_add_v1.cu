// C++ headers
#include <random>
#include <chrono>

// C headers
#include <cstdio>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>

void vec_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

__global__ void vec_add_v1(float *a, float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return ;
    c[idx] = a[idx] + b[idx];
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> size_dist(1000'0000, 2000'0000);
    std::uniform_real_distribution<float> val_dist(0.0f, 10.0f);
    
    int n = size_dist(gen);
    size_t bytes = n * sizeof(float);
    
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c_cpu = new float[n];
    float *h_c_gpu = new float[n];
    
    for (int i = 0; i < n; i++) {
        h_a[i] = val_dist(gen);
        h_b[i] = val_dist(gen);
    }
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vec_add(h_a, h_b, h_c_cpu, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    
    cudaEventRecord(gpu_start, 0);
    vec_add_v1<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(gpu_end, 0);
    cudaEventSynchronize(gpu_end);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_end);
    
    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (std::fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-4) {
            correct = false;
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            break;
        }
    }
    
    printf("Vector size: %d\n", n);
    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    if (correct)
        printf("Result verification: PASS\n");
    else
        printf("Result verification: FAIL\n");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    
    return 0;
}