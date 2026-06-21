/*
    nvcc -o async_api async_api.cu
    async_api.exe
*/

#include <chrono>

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define CUDA_CHECK(func) do {                                             \
    cudaError_t err = func;                                               \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "[CUDA ERROR] %s: %d: %s.\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void kernel_func(int *x, int n, int dx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return ;
    x[idx] = x[idx] + dx;
}

bool check_x(int *x, const int n, const int expected) {
    for (int i = 0; i < n; i++) {
        if (x[i] != expected) {
            printf("[ERROR]: x[%d] = %d, expected value = %d.\n", i, x[i], expected);
            return false;
        }
    }
    return true;
}

int main() {
    
    printf("\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("No device found!\n");
        return EXIT_FAILURE;
    }

    int device_id = 0;
    cudaDeviceProp device_props;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));
    printf("CUDA device: %s.\n", device_props.name);

    int n      = 256 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value  = 26;

    int *h_a = nullptr;
    CUDA_CHECK(cudaMallocHost((void **)&h_a, nbytes));
    memset(h_a, 0x00, nbytes);

    int *d_a = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_a, nbytes));
    CUDA_CHECK(cudaMemset(d_a, 0xff, nbytes));

    int threads = 256;
    int blocks  = CEIL_DIV(n, threads);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());

    auto cpu_start = std::chrono::high_resolution_clock::now();

    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice, 0);
    kernel_func <<<blocks, threads, 0, 0>>> (d_a, n, value);
    cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);

    auto cpu_stop = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaProfilerStop());

    unsigned long counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    float gpu_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    std::chrono::duration<float, std::milli> cpu_time = cpu_stop - cpu_start;

    printf("\n");
    printf("time spent executing by the GPU: %.2f ms.\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f ms.\n", cpu_time.count());
    printf("CPU executed %lu iterations while waiting for GPU to finish.\n", counter);

    printf("\n");
    bool is_pass = check_x(h_a, n, value);
    printf("Verification: %s.\n", is_pass? "PASS": "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFree(d_a));

    return is_pass? EXIT_SUCCESS: EXIT_FAILURE;
}
