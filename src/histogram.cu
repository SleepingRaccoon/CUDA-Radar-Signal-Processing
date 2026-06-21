#include "histogram.cuh"

void histogram_cpu(int *a, int *y, int N) {
    for (int i = 0; i < N; i++)
        y[a[i]]++;
}

__global__ void histogram_i32(int *a, int *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(&y[a[idx]], 1);
}

__global__ void histogram_i32x4(int *a, int *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 3 < N) {
        int4 tmp_a = INT4(a[idx]);
        atomicAdd(&y[tmp_a.x], 1);
        atomicAdd(&y[tmp_a.y], 1);
        atomicAdd(&y[tmp_a.z], 1);
        atomicAdd(&y[tmp_a.w], 1);        
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++)
            atomicAdd(&y[a[idx + i]], 1);
    }
}