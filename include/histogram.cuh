#pragma once

#include "macros.cuh"

void histogram_cpu(int *a, int *y, int N);

__global__ void histogram_i32(int *a, int *y, int N);

__global__ void histogram_i32x4(int *a, int *y, int N);



