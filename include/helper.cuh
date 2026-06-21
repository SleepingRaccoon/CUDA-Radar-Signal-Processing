#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(err));                                       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

template<typename... Args>
static float KernelTime(void (*kernel)(Args...), dim3 grid, dim3 block,
                   int reps, unsigned int shmem, cudaStream_t stream,
                   Args... args)
{
    cudaEvent_t st, en;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&en));
    CUDA_CHECK(cudaEventRecord(st));
    for (int i = 0; i < reps; i++)
        kernel<<<grid, block, shmem, stream>>>(args...);
    CUDA_CHECK(cudaEventRecord(en));
    CUDA_CHECK(cudaEventSynchronize(en));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, en));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(en));
    return ms / reps;
}
