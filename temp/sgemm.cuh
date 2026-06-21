#pragma once

#include "macros.cuh"

void sgemm_cpu(const float *A, const float *B, float *C,
                const int M, const int K, const int N,
                const float alpha, const float beta);

__global__ void sgemm_v1(const float *A, const float *B, float *C,
                            const int M, const int K, const int N,
                            const float alpha, const float beta);

template <const int BM, const int BK, const int BN>
__device__ void sgemm_v2(const float *A, const float *B, float *C,
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

        // [BUG FIXED] v2 line 62: 原为 for (int kk = 0; kk < BK; k++)
        // 循环变量声明为 kk，但递增了 k（外部循环变量）
        // → 导致死循环，kernel 永远不会退出
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

    constexpr int TY = BM / TM;         // block y 方向线程数
    constexpr int TX = BN / TN;         // block x 方向线程数
    constexpr int N_THREADS = TY * TX;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;               // [0, TY)
    int tx = threadIdx.x;               // [0, TX)

    // [BUG FIXED] v3 lines 93-94: 原为 ty * TY / tx * TX
    // 错误：ty 是线程在 block 中的 y 索引（范围 0..TY-1），
    //       每个线程负责 TM 行，正确偏移应为 ty * TM
    // 同理 tx 的正确偏移应为 tx * TN
    // ty*TY 和 ty*TM 仅在 BM=TM² 时偶然相等，否则完全错误
    int base_row = by * BM + ty * TM;
    int base_col = bx * BN + tx * TN;

    int tid = ty * TX + tx;

    __shared__ float sh_A[BM][BK];
    __shared__ float sh_B[BK][BN];

    float sum[TM][TN] = {};

    for (int bk = 0; bk < K; bk += BK) {

        // [BUG FIXED] v3 lines 106-107: 原为 r = tid / BK; c = tid % BK;
        // 错误：循环变量是 idx，但内部用了 tid 计算位置。
        // 当 N_THREADS < BM*BK 时，idx 变化但位置不变，
        // 导致一部分元素重复加载，另一部分从未加载。
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

        // [BUG FIXED] v3 lines 116-117: 原有 3 个错误
        //   ① 循环上界 BK*BM → 应为 BK*BN（sB 是 BK×BN 不是 BK×BM）
        //   ② 内部用 tid 而非 idx（同上，重复加载）
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

    // [BUG FIXED] v3 lines 143-152: C 写回原在 K 循环内部（第 153 行 } 闭合 K 循环）
    // 错误：sum 在 K 循环外累积，但每轮 K 迭代都写回 C，
    //       C = alpha*sum + beta*C_old 中 C_old 被上一轮覆盖，
    //       导致 beta≠0 时结果错误
    // 修正：将写回移到 K 循环之后，此时 sum 已是完整累积结果
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

    int ty = threadIdx.y;               // [0, TY)
    int tx = threadIdx.x;               // [0, TX)

    constexpr int TY = BM / TM;
    constexpr int TX = BN / TN;
    constexpr int N_THREADS = TY * TX;

    // [BUG FIXED] v4 line 178: 原为 tid = ty * TY + tx
    // 错误：threadIdx.y 范围是 [0, TY)，threadIdx.x 范围是 [0, TX)
    //       正确的一维映射应为 ty * TX + tx（列主序）
    //       原式 ty*TY + tx 在 TY ≠ TX 时导致 tid 映射错误
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

        // [BUG FIXED] v4 line 196: 原为 gc = by * BN + c
        // 错误：by 是 block 的 y（行）索引，但 B 的列号应由 block 的 x（列）索引计算
        //       bx 才是 blockIdx.x，正确的列号 = bx * BN + c
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

        // [BUG FIXED] v4 lines 235, 238: 原为 sh_A[ty * TY + i] / sh_B[kk][tx * TX + j]
        // 错误：ty 是线程索引，每个线程负责 TM 行，正确偏移为 ty * TM
        //       同理 tx 的正确偏移为 tx * TN
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

    // [BUG FIXED] v4 lines 244-268:
    //   ① 写回索引中 ty*TY/tx*TX → 改为 ty*TM/tx*TN（同上）
    //   ② LD_FLOAT4(c[...]) 中 c → C（编译错误，C 是参数名）
    //   ③ 写回从 K 循环中移出（与 v3 相同的问题）
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
                // [BUG FIXED] v4 line 256: 原为 LD_FLOAT4(c[r * N + c0])
                // 参数名是 C（大写），c 未定义导致编译错误
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
