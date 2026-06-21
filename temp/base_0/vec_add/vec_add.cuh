#include <iostream>

#include <cuda_runtime.h>

#define FLOAT4(x) (*reinterpret_cast<const float4 *>(&(x)))

__global__ void vec_add_v1(const float * __restrict__ a, 
                            const float * __restrict__ b, 
                            int n,
                            float * __restrict__ c);

__global__ void vec_add_v2(const float * __restrict__ a, 
                            const float * __restrict__ b, 
                            int n,
                            float * __restrict__ c);

__global__ void vec_add_v3(const float * __restrict__ a, 
                            const float * __restrict__ b, 
                            int n,
                            float * __restrict__ c);

