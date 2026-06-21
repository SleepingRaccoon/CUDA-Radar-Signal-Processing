#include "elementwise_add.cuh"

__global__ void elementwise_add_f32(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_f32x4(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 3 < N) {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;        
        FLOAT4(c[idx]) = tmp_c;
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++) 
            c[idx + i] = a[idx + i] + b[idx + i];
    }
}

__global__ void elementwise_add_f16(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx]; 
}

__global__ void elementwise_add_f16x2(half *a, half *b, half *c, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 1 < N) {
        half2 tmp_a = HALF2(a[idx]);
        half2 tmp_b = HALF2(b[idx]);
        half2 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        HALF2(c[idx]) = tmp_c;
    }
    else if (idx < N)
        c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_f16x8_v1(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        half2 tmp_a0 = HALF2(a[idx + 0]);
        half2 tmp_a2 = HALF2(a[idx + 2]);
        half2 tmp_a4 = HALF2(a[idx + 4]);
        half2 tmp_a6 = HALF2(a[idx + 6]);

        half2 tmp_b0 = HALF2(b[idx + 0]);
        half2 tmp_b2 = HALF2(b[idx + 2]);
        half2 tmp_b4 = HALF2(b[idx + 4]);
        half2 tmp_b6 = HALF2(b[idx + 6]);

        half2 tmp_c0 = __hadd2(tmp_a0, tmp_b0);
        half2 tmp_c2 = __hadd2(tmp_a2, tmp_b2);
        half2 tmp_c4 = __hadd2(tmp_a4, tmp_b4);
        half2 tmp_c6 = __hadd2(tmp_a6, tmp_b6);
        
        HALF2(c[idx + 0]) = tmp_c0;
        HALF2(c[idx + 2]) = tmp_c2;
        HALF2(c[idx + 4]) = tmp_c4;
        HALF2(c[idx + 6]) = tmp_c6;
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++)
            c[idx + i] = a[idx + i] + b[idx + i]; 
    }
}

__global__ void elementwise_add_f16x8_v2(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        half tmp_a[8], tmp_b[8], tmp_c[8];
        LD_ST_128BITS(tmp_a[0]) = LD_ST_128BITS(a[idx]);
        LD_ST_128BITS(tmp_b[0]) = LD_ST_128BITS(b[idx]);

        #pragma unroll
        for (int i = 0; i < 8; i += 2)
            HALF2(tmp_c[i]) = __hadd2(HALF2(tmp_a[i]), HALF2(tmp_b[i]));
        
        LD_ST_128BITS(c[idx]) = LD_ST_128BITS(tmp_c[0]);
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++)
            c[idx + i] = a[idx + i] + b[idx + i];
    }
}
