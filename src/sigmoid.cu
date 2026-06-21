#include "sigmoid.cuh"

void sigmoid_f32_cpu(float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        float v = fminf(fmaxf(x[i], F32_EXP_MIN_X), F32_EXP_MAX_X);
        y[i] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_f32(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = fminf(fmaxf(x[idx], F32_EXP_MIN_X), F32_EXP_MAX_X);
        y[idx] = 1.0f / (1.0f + expf(-v));        
    }
}

__global__ void sigmoid_f32x4(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 3 < N) {
        float4 tmp = FLOAT4(x[idx]);
        
        tmp.x = fminf(fmaxf(tmp.x, F32_EXP_MIN_X), F32_EXP_MAX_X);
        tmp.x = 1.0f / (1.0f + expf(-tmp.x));
        
        tmp.y = fminf(fmaxf(tmp.y, F32_EXP_MIN_X), F32_EXP_MAX_X);
        tmp.y = 1.0f / (1.0f + expf(-tmp.y));
        
        tmp.z = fminf(fmaxf(tmp.z, F32_EXP_MIN_X), F32_EXP_MAX_X);
        tmp.z = 1.0f / (1.0f + expf(-tmp.z));
        
        tmp.w = fminf(fmaxf(tmp.w, F32_EXP_MIN_X), F32_EXP_MAX_X);
        tmp.w = 1.0f / (1.0f + expf(-tmp.w));
        
        FLOAT4(y[idx]) = tmp;
    }    
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++) {
            float v = fminf(fmaxf(x[idx + i], F32_EXP_MIN_X), F32_EXP_MAX_X);
            y[idx + i] = 1.0f / (1.0f + expf(-v));
        }
    }
}

void sigmoid_f16_cpu(half *x, half *y, int N) {
    for (int i = 0; i < N; i++) {
        half v = __hmin(__hmax(x[i], F16_EXP_MIN_X), F16_EXP_MAX_X);
        y[i] = __float2half(1.0f / (1.0f + expf(-__half2float(v))));
    }
}

__global__ void sigmoid_f16(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        half v = __hmin(__hmax(x[idx], F16_EXP_MIN_X), F16_EXP_MAX_X);
        y[idx] = F16_ONE / (F16_ONE + hexp(-v));        
    }
}

__global__ void sigmoid_f16x2(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 1 < N) {
        half2 tmp = HALF2(x[idx]);
        
        tmp.x = __hmin(__hmax(tmp.x, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp.x = F16_ONE / (F16_ONE + hexp(-tmp.x));
        
        tmp.y = __hmin(__hmax(tmp.y, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp.y = F16_ONE / (F16_ONE + hexp(-tmp.y));
        
        HALF2(y[idx]) = tmp;
    }
    else if (idx < N) {
        half v = __hmin(__hmax(x[idx], F16_EXP_MIN_X), F16_EXP_MAX_X);
        y[idx] = F16_ONE / (F16_ONE + hexp(-v));
    }
}

__global__ void sigmoid_f16x8_v1(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 7 < N) {
        half2 tmp_0 = HALF2(x[idx + 0]);
        half2 tmp_2 = HALF2(x[idx + 2]);
        half2 tmp_4 = HALF2(x[idx + 4]);
        half2 tmp_6 = HALF2(x[idx + 6]);        
        
        tmp_0.x = __hmin(__hmax(tmp_0.x, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_0.x = F16_ONE / (F16_ONE + hexp(-tmp_0.x));
        tmp_0.y = __hmin(__hmax(tmp_0.y, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_0.y = F16_ONE / (F16_ONE + hexp(-tmp_0.y));

        tmp_2.x = __hmin(__hmax(tmp_2.x, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_2.x = F16_ONE / (F16_ONE + hexp(-tmp_2.x));
        tmp_2.y = __hmin(__hmax(tmp_2.y, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_2.y = F16_ONE / (F16_ONE + hexp(-tmp_2.y));

        tmp_4.x = __hmin(__hmax(tmp_4.x, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_4.x = F16_ONE / (F16_ONE + hexp(-tmp_4.x));
        tmp_4.y = __hmin(__hmax(tmp_4.y, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_4.y = F16_ONE / (F16_ONE + hexp(-tmp_4.y));

        tmp_6.x = __hmin(__hmax(tmp_6.x, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_6.x = F16_ONE / (F16_ONE + hexp(-tmp_6.x));
        tmp_6.y = __hmin(__hmax(tmp_6.y, F16_EXP_MIN_X), F16_EXP_MAX_X);
        tmp_6.y = F16_ONE / (F16_ONE + hexp(-tmp_6.y));

        HALF2(y[idx + 0]) = tmp_0;
        HALF2(y[idx + 2]) = tmp_2;
        HALF2(y[idx + 4]) = tmp_4;
        HALF2(y[idx + 6]) = tmp_6;
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++) {
            half v = __hmin(__hmax(x[idx + i], F16_EXP_MIN_X), F16_EXP_MAX_X);
            y[idx + i] = F16_ONE / (F16_ONE + hexp(-v));
        }
    }    
}

__global__ void sigmoid_f16x8_v2(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 7 < N) {
        half tmp[8];
        LD_ST_128BITS(tmp[0]) = LD_ST_128BITS(x[idx]);
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            tmp[i] = __hmin(__hmax(tmp[i], F16_EXP_MIN_X), F16_EXP_MAX_X);
            tmp[i] = F16_ONE / (F16_ONE + hexp(-tmp[i]));
        }
        
        LD_ST_128BITS(y[idx]) = LD_ST_128BITS(tmp[0]);
    }
    else if (idx < N) {
        #pragma unroll
        for (int i = 0; idx + i < N; i++) {
            half v = __hmin(__hmax(x[idx + i], F16_EXP_MIN_X), F16_EXP_MAX_X);
            y[idx + i] = F16_ONE / (F16_ONE + hexp(-v));
        }
    }  
}
