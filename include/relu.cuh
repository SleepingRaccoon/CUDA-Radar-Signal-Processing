#pragma once

#include <cuda_fp16.h>
#include <cmath>

#include "macros.cuh"

void relu_f32_cpu(float *x, float *y, int N);

__global__ void relu_f32(float *x, float *y, int N);

__global__ void relu_f32x4(float *x, float *y, int N);

void relu_f16_cpu(half *x, half *y, int N);

__global__ void relu_f16(half *x, half *y, int N);

__global__ void relu_f16x2(half *x, half *y, int N);

__global__ void relu_f16x8_v1(half *x, half *y, int N);

__global__ void relu_f16x8_v2(half *x, half *y, int N);
