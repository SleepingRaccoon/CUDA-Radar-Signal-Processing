#pragma once 

#include <cuda_fp16.h>

#include "macros.cuh"

template <typename T> void elementwise_add_cpu(T *a, T *b, T *c, int N) {
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

__global__ void elementwise_add_f32(float *a, float *b, float *c, int N);

__global__ void elementwise_add_f32x4(float *a, float *b, float *c, int N);

__global__ void elementwise_add_f16(half *a, half *b, half *c, int N);

__global__ void elementwise_add_f16x2(half *a, half *b, half *c, int N);

__global__ void elementwise_add_f16x8_v1(half *a, half *b, half *c, int N);

__global__ void elementwise_add_f16x8_v2(half *a, half *b, half *c, int N);


