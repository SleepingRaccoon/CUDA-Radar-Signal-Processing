#include "vec_add.cuh"

__global__ void vec_add_v1(const float * __restrict__ a, 
                            const float * __restrict__ b, 
                            int n,
                            float * __restrict__ c)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return ;
    c[idx] = a[idx] + b[idx];
}

__global__ void vec_add_v2(const float * __restrict__ a, 
                            const float * __restrict__ b, 
                            int n,
                            float * __restrict__ c)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}

__global__ void vec_add_v3(const float * __restrict__ a, 
                            const float * __restrict__ b, 
                            int n,
                            float * __restrict__ c)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int remainning = n - idx;
    if (remainning >= 3) {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;        
    }
    else if (remainning >= 0){
#pragma unroll
        for (int i = 0; i < remainning; i++)
            c[idx + i] = a[idx + i] + b[idx + i];
    }
}