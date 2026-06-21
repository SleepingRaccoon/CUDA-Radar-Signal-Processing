#pragma once

#include "macros.cuh"

void sgemm_cpu(const float *A, const float *B, float *C, 
                const int M, const int K, const int N, 
                const float alpha, const float beta);

__global__ void sgemm_v1(const float *A, const float *B, float *C, 
                            const int M, const int K, const int N, 
                            const float alpha, const float beta);

template <const int BM, const int BK, const int BN>
__global__ void sgemm_v2(const float *A, const float *B, float *C, 
                            const int M, const int K, const int N, 
                            const float alpha, const float beta)
{
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int tid = ty * BN + tx;

    int row = by * BM + ty;
    int col = bx * BN + tx;
    
    constexpr int N_THREADS = BM * BN;

    __shared__ float sh_A[BM][BK];
    __shared__ float sh_B[BK][BN];

    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += BK) {

        for (int idx = tid; idx < BM * BK; idx += N_THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            int gr = by * BM + r;
            int gc = bk + c;
            if (gr < M && gc < K) 
                sh_A[r][c] = A[gr * K + gc];
            else 
                sh_A[r][c] = 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += N_THREADS) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = bk + r;
            int gc = bx * BN + c;
            if (gr < K && gc < N) 
                sh_B[r][c] = B[gr * N + gc];
            else 
                sh_B[r][c] = 0.0f;
        }        

        __syncthreads();

        for (int kk = 0; kk < BK; kk++)
            sum += sh_A[ty][kk] * sh_B[kk][tx];
            
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
}


template <const int BM, const int BK, const int BN, 
            const int TM, const int TN>
__global__ void sgemm_v3(const float *A, const float *B, float *C, 
                            const int M, const int K, const int N, 
                            const float alpha, const float beta)
{

    static_assert(BM % TM == 0, "[SGEMM ERROR] BM % TM != 0.\n");
    static_assert(BN % TN == 0, "[SGEMM ERROR] BN % TN != 0.\n");

    constexpr int TY = BM / TM;
    constexpr int TX = BN / TN;
    constexpr int N_THREADS = TY * TX;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int base_row = by * BM + ty * TM;
    int base_col = bx * BN + tx * TN;

    int tid = ty * TX + tx;

    __shared__ float sh_A[BM][BK];
    __shared__ float sh_B[BK][BN];

    float sum[TM][TN] = {};

    for (int bk = 0; bk < K; bk += BK) {
        
        for (int idx = tid; idx < BM * BK; idx += N_THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            int gr = by * BM + r;
            int gc = bk + c;
            if (gr < M && gc < K)
                sh_A[r][c] = A[gr * K + gc];
            else 
                sh_A[r][c] = 0.0f; 
        }

        for (int idx = tid; idx < BK * BN; idx += N_THREADS) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = bk + r;
            int gc = bx * BN + c;
            if (gr < K && gc < N)
                sh_B[r][c] = B[gr * N + gc];
            else 
                sh_B[r][c] = 0.0f; 
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                float va = sh_A[ty * TM + i][kk];
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    sum[i][j] += va * sh_B[kk][tx * TN + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            if (base_row + i < M && base_col + j < N) {
                C[(base_row + i) * N + (base_col + j)] = 
                    alpha * sum[i][j] + beta * C[(base_row + i) * N + (base_col + j)];
            }
        }            
    }
}

template <const int BM, const int BK, const int BN, 
            const int TM, const int TN>
__global__ void sgemm_v4(const float *A, const float *B, float *C, 
                            const int M, const int K, const int N, 
                            const float alpha, const float beta)
{
    static_assert(BM % TM == 0, "[SGEMM ERROR] BM % TM != 0.\n");
    static_assert(BN % TN == 0, "[SGEMM ERROR] BN % TN != 0.\n");
    static_assert(BK % 4 == 0, "[SGEMM ERROR] BK % 4 != 0.\n");
    static_assert(BN % 4 == 0, "[SGEMM ERROR] BN % 4 != 0.\n");
    static_assert(TN % 4 == 0, "[SGEMM ERROR] TN % 4 != 0.\n");

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    constexpr int TY = BM / TM;
    constexpr int TX = BN / TN;
    constexpr int N_THREADS = TY * TX;

    int tid = ty * TX + tx;
    
    constexpr int F4 = 4;
    constexpr int N_A = BM * BK / F4;
    constexpr int N_B = BK * BN / F4;

    __shared__ float sh_A[BM][BK];
    __shared__ float sh_B[BK][BN];

    float sum[TM][TN] = {};

    for (int bk = 0; bk < K; bk += BK) {
        
        for (int idx = tid; idx < N_A; idx += N_THREADS) {
            int r = idx / (BK / F4);
            int c = idx % (BK / F4) * 4;
            int gr = by * BM + r;
            int gc = bk + c;
            if (gr < M && gc + 3 < K)
                ST_FLOAT4(sh_A[r][c]) = LD_FLOAT4(A[gr * K + gc]);
            else if (gr < M && gc < K) {
                #pragma unroll
                for (int f = 0; f < F4; f++)
                    sh_A[r][c + f] = (gc + f < K)? A[gr * K + gc + f]: 0.0f;
            }
            else {
                #pragma unroll
                for (int f = 0; f < F4; f++)
                    sh_A[r][c + f] = 0.0f;                
            }
        }

        for (int idx = tid; idx < N_B; idx += N_THREADS) {
            int r = idx / (BN / F4);
            int c = idx % (BN / F4) * 4;
            int gr = bk + r;
            int gc = bx * BN + c;
            if (gr < K && gc + 3 < N)
                ST_FLOAT4(sh_B[r][c]) = LD_FLOAT4(B[gr * N + gc]);
            else if (gr < K && gc < N) {
                #pragma unroll
                for (int f = 0; f < F4; f++)
                    sh_B[r][c + f] = (gc + f < N)? B[gr * N + gc + f]: 0.0f;
            }
            else {
                #pragma unroll
                for (int f = 0; f < F4; f++)
                    sh_B[r][c + f] = 0.0f;                
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                float va = sh_A[ty * TM + i][kk];
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    sum[i][j] += va * sh_B[kk][tx * TN + j];
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int r = by * BM + ty * TM + i;
        if (r >= M)
            break;
        int c0 = bx * BN + tx * TN;
        if (c0 + TN - 1 < N) {
            for (int j = 0; j < TN; j += F4) {
                float4 v4;
                v4.x = sum[i][j + 0];
                v4.y = sum[i][j + 1];
                v4.z = sum[i][j + 2];
                v4.w = sum[i][j + 3];
                const float4 c4 = LD_FLOAT4(C[r * N + c0]);
                ST_FLOAT4(C[r * N + c0]) = 
                    make_float4(alpha * v4.x, alpha * v4.y, alpha * v4.z, alpha * v4.w) +
                    make_float4(beta * c4.x, beta * c4.y, beta * c4.z, beta * c4.w);    
            }            
        }
        else {
            for (int j = 0; j < TN; j++) {
                if (c0 + j < N)
                    C[r * N + (c0 + j)] = alpha * sum[i][j] + beta * C[r * N + (c0 + j)];
            }
        }
    }
}