#include "sgemm.cuh"

void sgemm_cpu(const float *A, const float *B, float *C, 
                const int M, const int K, const int N, 
                const float alpha, const float beta)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[ROW_MAJOR(i, k, K)] * B[ROW_MAJOR(k, j, N)];
            C[ROW_MAJOR(i, j, N)] = alpha * s + beta * C[ROW_MAJOR(i, j, N)];         
        }
    }
}

__global__ void sgemm_v1(const float *A, const float *B, float *C, 
                            const int M, const int K, const int N, 
                            const float alpha, const float beta) 
{
    int g_ty = blockDim.y * blockIdx.y + threadIdx.y;
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_ty < M && g_tx < N) {
        float s = 0.0f;
        for (int k = 0; k < K; k++)
            s += A[ROW_MAJOR(g_ty, k, K)] * B[ROW_MAJOR(k, g_tx, N)];
        C[ROW_MAJOR(g_ty, g_tx, N)] = alpha * s + beta * C[ROW_MAJOR(g_ty, g_tx, N)];  
    }
}

