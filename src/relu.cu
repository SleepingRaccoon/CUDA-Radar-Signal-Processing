#include "relu.cuh"

void relu_f32_cpu(float *x, float *y, int N) {
    for (int i = 0; i < N; i++)
        y[i] = fmaxf(x[i], 0.0f);
}

__global__ void relu_f32(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        y[idx] = fmaxf(x[idx], 0.0f);        
}

__global__ void relu_f32x4(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 3 < N) {
        float4 tmp = FLOAT4(x[idx]);
        
        tmp.x = fmaxf(tmp.x, 0.0f);
        tmp.y = fmaxf(tmp.y, 0.0f);
        tmp.z = fmaxf(tmp.z, 0.0f);
        tmp.w = fmaxf(tmp.w, 0.0f);
        
        FLOAT4(y[idx]) = tmp;
    }    
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++)
            y[idx + i] = fmaxf(x[idx + i], 0.0f);
    }
}

void relu_f16_cpu(half *x, half *y, int N) {
    for (int i = 0; i < N; i++) 
        y[i] = __hmax(x[i], __float2half(0.0f));
}

__global__ void relu_f16(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        y[idx] = __hmax(x[idx], __float2half(0.0f));       
}

__global__ void relu_f16x2(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 1 < N) {
        half2 tmp = HALF2(x[idx]);
        tmp.x = __hmax(tmp.x, __float2half(0.0f));
        tmp.y = __hmax(tmp.y, __float2half(0.0f));
        HALF2(y[idx]) = tmp;
    }
    else if (idx < N)
        y[idx] = __hmax(x[idx], __float2half(0.0f));
}

__global__ void relu_f16x8_v1(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        half2 tmp_0 = HALF2(x[idx + 0]);
        half2 tmp_2 = HALF2(x[idx + 2]);
        half2 tmp_4 = HALF2(x[idx + 4]);
        half2 tmp_6 = HALF2(x[idx + 6]);        
        
        tmp_0.x = __hmax(tmp_0.x, __float2half(0.0f));
        tmp_0.y = __hmax(tmp_0.y, __float2half(0.0f));

        tmp_2.x = __hmax(tmp_2.x, __float2half(0.0f));
        tmp_2.y = __hmax(tmp_2.y, __float2half(0.0f));

        tmp_4.x = __hmax(tmp_4.x, __float2half(0.0f));
        tmp_4.y = __hmax(tmp_4.y, __float2half(0.0f));

        tmp_6.x = __hmax(tmp_6.x, __float2half(0.0f));
        tmp_6.y = __hmax(tmp_6.y, __float2half(0.0f));

        HALF2(y[idx + 0]) = tmp_0;
        HALF2(y[idx + 2]) = tmp_2;
        HALF2(y[idx + 4]) = tmp_4;
        HALF2(y[idx + 6]) = tmp_6;
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++)
            y[idx + i] = __hmax(x[idx + i], __float2half(0.0f));
    }    
}

__global__ void relu_f16x8_v2(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        half2 tmp[4];
        LD_ST_128BITS(tmp[0]) = LD_ST_128BITS(x[idx]);
        half2 zero2 = __float2half2_rn(0.0f);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            tmp[i] = __hmax2(tmp[i], zero2);
        LD_ST_128BITS(y[idx]) = LD_ST_128BITS(tmp[0]);
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++)
            y[idx + i] = __hmax(x[idx + i], __float2half(0.0f));
    }  
}
