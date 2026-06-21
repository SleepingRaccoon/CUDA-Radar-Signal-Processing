#include "macros.cuh"

__host__ void sgeva_cpu(const float * __restrict__ a,
                            const float * __restrict__ b,
                            const int n,
                            float * __restrict__ c);

__global__ void sgeva_v11(const float * __restrict__ a,
                            const float * __restrict__ b,
                            const int n,
                            float * __restrict__ c);

__global__ void sgeva_v12(const float * __restrict__ a,
                            const float * __restrict__ b,
                            const int n,
                            float * __restrict__ c);

__global__ void sgeva_v21(const float * __restrict__ a,
                            const float * __restrict__ b,
                            const int n,
                            float * __restrict__ c);

__global__ void sgeva_v22(const float * __restrict__ a,
                            const float * __restrict__ b,
                            const int n,
                            float * __restrict__ c);
