/*
 * sgemm_v5.cu — 在 sgemm_V2 基础上进一步优化 (通用参数版本)
 *
 * V2 已经做的优化:
 *   ① s_a 转置布局 s_a[BK][BM] → 计算阶段连续读
 *   ② 寄存器预载：s_a/s_b 先读到 r_comp_a/b 再计算
 *   ③ float4 向量化访存
 *   ④ C 写回拆分
 *
 * v5 新增:
 *   ① padding: s_a[BK][BM+1], s_b[BK][BN+1] — 消解 bank conflict
 *   ② stride loop + flat tid — 线程利用率 100%，参数通用
 *   ③ 边界处理: 任意 M/N/K 均可
 *   ④ 全部索引基于模板参数推导，BM/BK/BN/TM/TN 自由调节
 *
 * 约束:
 *   BM % TM == 0, BN % TN == 0
 *   BK % 4 == 0, BN % 4 == 0, TM % 4 == 0, TN % 4 == 0
 */

#include <cuda_runtime.h>
#include "../include/macros.cuh"

#define F4 4

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K)
{
    /* ── 编译期约束 ── */
    static_assert(BM % TM == 0, "BM % TM != 0");
    static_assert(BN % TN == 0, "BN % TN != 0");
    static_assert(BK % F4 == 0, "BK % 4 != 0");
    static_assert(BN % F4 == 0, "BN % 4 != 0");
    static_assert(TM % F4 == 0, "TM % 4 != 0");
    static_assert(TN % F4 == 0, "TN % 4 != 0");

    /* ── 线程/block 维度 ── */
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int N_THREADS = TX * TY;

    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int tx  = threadIdx.x;          // [0, TX)
    int ty  = threadIdx.y;          // [0, TY)
    int tid = ty * TX + tx;         // flat

    /* ── 共享内存 (padding) ── */
    constexpr int BM_PAD = BM + 1;
    constexpr int BN_PAD = BN + 1;
    __shared__ float s_a[BK][BM_PAD];
    __shared__ float s_b[BK][BN_PAD];

    /* ── 寄存器 ── */
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {};

    /* ── 当前线程的 C 块起始坐标 ── */
    int c_row = by * BM + ty * TM;
    int c_col = bx * BN + tx * TN;

    /* ── 需要加载的 float4 组数 ── */
    constexpr int SA_F4 = BM * BK / F4;
    constexpr int SB_F4 = BK * BN / F4;

    /* ── K 主循环 ── */
    for (int bk = 0; bk < K; bk += BK) {

        /* ══ 加载 A → s_a (转置: s_a[BK][BM]) ══ */
        for (int idx = tid; idx < SA_F4; idx += N_THREADS) {
            int m  = idx / (BK / F4);        // s_a 的 BM 列
            int ks = (idx % (BK / F4)) * F4; // s_a 的 BK 行起点

            int gr = by * BM + m;
            int gc = bk + ks;

            if (gr < M) {
                if (gc + 3 < K) {
                    float4 tmp = LD_FLOAT4(A[ROW_MAJOR(gr, gc, K)]);
                    s_a[ks    ][m] = tmp.x;
                    s_a[ks + 1][m] = tmp.y;
                    s_a[ks + 2][m] = tmp.z;
                    s_a[ks + 3][m] = tmp.w;
                } else {
                    #pragma unroll
                    for (int f = 0; f < F4; f++) {
                        int kk = gc + f;
                        s_a[ks + f][m] = (kk < K) ? A[ROW_MAJOR(gr, kk, K)] : 0.0f;
                    }
                }
            } else {
                #pragma unroll
                for (int f = 0; f < F4; f++) s_a[ks + f][m] = 0.0f;
            }
        }

        /* ══ 加载 B → s_b ══ */
        for (int idx = tid; idx < SB_F4; idx += N_THREADS) {
            int k  = idx / (BN / F4);        // s_b 行
            int ns = (idx % (BN / F4)) * F4; // s_b 列起点

            int gr = bk + k;
            int gc = bx * BN + ns;

            if (gr < K) {
                if (gc + 3 < N) {
                    ST_FLOAT4(s_b[k][ns]) = LD_FLOAT4(B[ROW_MAJOR(gr, gc, N)]);
                } else {
                    #pragma unroll
                    for (int f = 0; f < F4; f++) {
                        int nn = gc + f;
                        s_b[k][ns + f] = (nn < N) ? B[ROW_MAJOR(gr, nn, N)] : 0.0f;
                    }
                }
            } else {
                #pragma unroll
                for (int f = 0; f < F4; f++) s_b[k][ns + f] = 0.0f;
            }
        }

        __syncthreads();

        /* ══ 寄存器预载 + 计算 ══ */
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            /* 从 shared memory → 寄存器 */
            #pragma unroll
            for (int f = 0; f < TM; f += F4)
                ST_FLOAT4(r_comp_a[f]) = ST_FLOAT4(s_a[tk][ty * TM + f]);

            #pragma unroll
            for (int f = 0; f < TN; f += F4)
                ST_FLOAT4(r_comp_b[f]) = ST_FLOAT4(s_b[tk][tx * TN + f]);

            /* 全寄存器 FMA */
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }

        __syncthreads();
    }

    /* ══ 写回 C (float4 + 边界) ══ */
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = c_row + i;
        if (r >= M) break;

        if (c_col + TN - 1 < N) {
            #pragma unroll
            for (int j = 0; j < TN; j += F4) {
                float4 v;
                v.x = r_c[i][j + 0];
                v.y = r_c[i][j + 1];
                v.z = r_c[i][j + 2];
                v.w = r_c[i][j + 3];
                ST_FLOAT4(C[ROW_MAJOR(r, c_col + j, N)]) = v;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < TN; j++)
                if (c_col + j < N)
                    C[ROW_MAJOR(r, c_col + j, N)] = r_c[i][j];
        }
    }
}

#undef F4
