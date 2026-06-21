#include "reduce.cuh"

__host__ void reduce_cpu(const float *d_x, const int n, float *d_y) {
    *d_y = 0.0f;
    for (int i = 0; i < n; i++)
        *d_y += d_x[i];
}

__global__ void reduce_global(float *d_x, const int n, float *d_y) {
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    float *px = d_x + blockDim.x * blockIdx.x;
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            px[tid] += (g_tx + offset < n) ? px[tid + offset] : 0.0f;
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_y, (g_tx < n) ? px[tid] : 0.0f);
}

__global__ void reduce_shared(const float *d_x, const int n, float *d_y) {
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sh_mem[];
    sh_mem[tid] = (g_tx < n) ? d_x[g_tx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(d_y, sh_mem[0]);
}

__global__ void reduce_shared_stride(const float *d_x, const int n, float *d_y) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x)
        sum += d_x[i];

    extern __shared__ float sh_mem[];
    sh_mem[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(d_y, sh_mem[0]);
}

__global__ void reduce_syncwrap(const float *d_x, const int n, float *d_y) {
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sh_mem[];
    sh_mem[tid] = (g_tx < n) ? d_x[g_tx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= warpSize; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncthreads();
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncwarp();
    }

    if (tid == 0)
        atomicAdd(d_y, sh_mem[0]);
}

__global__ void reduce_shfl(const float *d_x, const int n, float *d_y) {
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sh_mem[];
    sh_mem[tid] = (g_tx < n) ? d_x[g_tx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float val = __shfl_down_sync(0xffffffff, sh_mem[tid], offset);
        if (tid < offset)
            sh_mem[tid] += val;
    }

    if (tid == 0)
        atomicAdd(d_y, sh_mem[0]);
}

