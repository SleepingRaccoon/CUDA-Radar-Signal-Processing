/*
 * sgemm_v6.cu — 在 v5 基础上引入流水线双缓冲 (double buffering)
 *
 * v5 已有的优化:
 *   ① s_a 转置布局 s_a[BK][BM]
 *   ② 寄存器预载 (r_comp_a/b)
 *   ∙ padding 消 bank conflict
 *   ∙ stride loop + flat tid 通用参数
 *   ∙ 边界处理
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ v6 新增: 双缓冲流水线化                                               │
 * │                                                                     │
 * │ V5 的执行时序:                                                       │
 * │   load → sync → compute → sync → load → sync → compute → sync ...  │
 * │                       ↑ latency 不能隐藏                             │
 * │                                                                     │
 * │ v6 的执行时序:                                                       │
 * │   [prologue] load tile 0                                             │
 * │   [loop bk=1..N-1]                                                  │
 * │     load(tile bk) → 与 compute(tile bk-1) 指令级交叠                   │
 * │     store(tile bk → 另一缓冲) → sync → ...                           │
 * │   [epilogue] compute last tile                                       │
 * │                       ↑ 全局加载延迟被计算的 FMA 掩盖                   │
 * │                                                                     │
 * │ 核心思路: s_a/s_b 各两份, 一组用于当前计算, 另一组同时正在被加载        │
 * │   s_a[2][BK][BM_PAD], s_b[2][BK][BN_PAD]                           │
 * │   每轮迭代: 算 buffer (bk-1)&1, 写 buffer bk&1                       │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * 约束 (与 v5 一致):
 *   BM % TM == 0, BN % TN == 0
 *   BK % 4 == 0, BN % 4 == 0, TM % 4 == 0, TN % 4 == 0
 */

#include <cuda_runtime.h>
#include "../include/macros.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v6(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K)
{
    /* ── 编译期约束 ── */
    static_assert(BM % TM == 0, "BM % TM != 0");
    static_assert(BN % TN == 0, "BN % TN != 0");
    static_assert(BK % 4 == 0,  "BK % 4 != 0");
    static_assert(BN % 4 == 0,  "BN % 4 != 0");
    static_assert(TM % 4 == 0,  "TM % 4 != 0");
    static_assert(TN % 4 == 0,  "TN % 4 != 0");

    /* ── 维度常数 ── */
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int N_THREADS = TX * TY;
    constexpr int F4 = 4;
    constexpr int BM_PAD = BM + 1;
    constexpr int BN_PAD = BN + 1;

    constexpr int SA_F4 = BM * BK / F4;       // s_a 总 float4 数
    constexpr int SB_F4 = BK * BN / F4;       // s_b 总 float4 数
    // 每个线程最多需要加载的 float4 数
    constexpr int SA_LOADS = (SA_F4 + N_THREADS - 1) / N_THREADS;
    constexpr int SB_LOADS = (SB_F4 + N_THREADS - 1) / N_THREADS;

    /* ── 索引 ── */
    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int tid = ty * TX + tx;

    /* ── [v6] 双缓冲 shared memory ── */
    __shared__ float s_a[2][BK][BM_PAD];
    __shared__ float s_b[2][BK][BN_PAD];

    /* ── 寄存器 ── */
    // [v6] 每线程的加载缓存: 先读到寄存器, 写入另一缓冲
    float r_a_load[SA_LOADS][F4];
    float r_b_load[SB_LOADS][F4];

    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {};

    int c_row_start = by * BM + ty * TM;
    int c_col_start = bx * BN + tx * TN;

    int num_tiles = (K + BK - 1) / BK;

    /* ============================================================
     * ══ 辅助: 加载 A → r_a_load (一次 tile) ══
     * ============================================================ */
    auto load_A_tile = [&](int bk) {
        for (int batch = 0; batch < SA_LOADS; batch++) {
            int idx = tid + batch * N_THREADS;
            if (idx >= SA_F4) continue;

            int m  = idx / (BK / F4);
            int ks = (idx % (BK / F4)) * F4;

            int gr = by * BM + m;
            int gc = bk * BK + ks;

            if (gr < M) {
                if (gc + 3 < K) {
                    float4 tmp = LD_FLOAT4(A[ROW_MAJOR(gr, gc, K)]);
                    r_a_load[batch][0] = tmp.x;
                    r_a_load[batch][1] = tmp.y;
                    r_a_load[batch][2] = tmp.z;
                    r_a_load[batch][3] = tmp.w;
                } else {
                    for (int f = 0; f < F4; f++) {
                        int kk = gc + f;
                        r_a_load[batch][f] = (kk < K) ? A[ROW_MAJOR(gr, kk, K)] : 0.0f;
                    }
                }
            } else {
                for (int f = 0; f < F4; f++)
                    r_a_load[batch][f] = 0.0f;
            }
        }
    };

    /* ============================================================
     * ══ 辅助: 加载 B → r_b_load (一次 tile) ══
     * ============================================================ */
    auto load_B_tile = [&](int bk) {
        for (int batch = 0; batch < SB_LOADS; batch++) {
            int idx = tid + batch * N_THREADS;
            if (idx >= SB_F4) continue;

            int k  = idx / (BN / F4);
            int ns = (idx % (BN / F4)) * F4;

            int gr = bk * BK + k;
            int gc = bx * BN + ns;

            if (gr < K) {
                if (gc + 3 < N) {
                    float4 tmp = LD_FLOAT4(B[ROW_MAJOR(gr, gc, N)]);
                    r_b_load[batch][0] = tmp.x;
                    r_b_load[batch][1] = tmp.y;
                    r_b_load[batch][2] = tmp.z;
                    r_b_load[batch][3] = tmp.w;
                } else {
                    for (int f = 0; f < F4; f++) {
                        int nn = gc + f;
                        r_b_load[batch][f] = (nn < N) ? B[ROW_MAJOR(gr, nn, N)] : 0.0f;
                    }
                }
            } else {
                for (int f = 0; f < F4; f++)
                    r_b_load[batch][f] = 0.0f;
            }
        }
    };

    /* ============================================================
     * ══ 辅助: r_a_load → s_a[buf] ══
     * ============================================================ */
    auto store_A_to_smem = [&](int buf) {
        for (int batch = 0; batch < SA_LOADS; batch++) {
            int idx = tid + batch * N_THREADS;
            if (idx >= SA_F4) continue;

            int m  = idx / (BK / F4);
            int ks = (idx % (BK / F4)) * F4;

            s_a[buf][ks    ][m] = r_a_load[batch][0];
            s_a[buf][ks + 1][m] = r_a_load[batch][1];
            s_a[buf][ks + 2][m] = r_a_load[batch][2];
            s_a[buf][ks + 3][m] = r_a_load[batch][3];
        }
    };

    /* ============================================================
     * ══ 辅助: r_b_load → s_b[buf] ══
     * ============================================================ */
    auto store_B_to_smem = [&](int buf) {
        for (int batch = 0; batch < SB_LOADS; batch++) {
            int idx = tid + batch * N_THREADS;
            if (idx >= SB_F4) continue;

            int k  = idx / (BN / F4);
            int ns = (idx % (BN / F4)) * F4;

            for (int f = 0; f < F4; f++)
                s_b[buf][k][ns + f] = r_b_load[batch][f];
        }
    };

    /* ============================================================
     * ══ 辅助: 从 s_a/b[buf] 计算 + 寄存器预载 ══
     * ============================================================ */
    auto compute_from = [&](int buf) {
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            /* 寄存器预载 */
            for (int f = 0; f < TM; f += F4)
                FLOAT4(r_comp_a[f]) = FLOAT4(s_a[buf][tk][ty * TM + f]);

            for (int f = 0; f < TN; f += F4)
                FLOAT4(r_comp_b[f]) = FLOAT4(s_b[buf][tk][tx * TN + f]);

            /* 全寄存器 FMA */
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
    };

    /* ════════════════════════════════════════════════════════════
     * ══ Phase 1: Prologue — 加载第 0 个 tile ══
     * ════════════════════════════════════════════════════════════ */
    load_A_tile(0);
    load_B_tile(0);
    store_A_to_smem(0);
    store_B_to_smem(0);
    __syncthreads();            // 保证 tile 0 数据全员可见

    /* ════════════════════════════════════════════════════════════
     * ══ Phase 2: Main Loop — bk = 1 .. num_tiles-1 ══
     * ══ 先发全局加载(→r_buf), 再计算上一批, 再存r_buf→smem ══
     * ════════════════════════════════════════════════════════════ */
    for (int bk = 1; bk < num_tiles; bk++) {
        int prev = (bk - 1) & 1;          // 存放 tile bk-1 的缓冲 → 本次计算
        int next = bk & 1;                // 存放 tile bk   的缓冲 → 本次写入

        // ① 从全局加载 tile bk (→ r_a_load / r_b_load)
        //   编译器可将这些全局加载指令与下面 compute_from 的 FMA 交叠
        load_A_tile(bk);
        load_B_tile(bk);

        // ② 计算 tile bk-1 (从 prev 缓冲)
        compute_from(prev);

        // ③ 将 r_a_load / r_b_load 写入 next 缓冲
        store_A_to_smem(next);
        store_B_to_smem(next);

        __syncthreads();                  // 保证 next 写入完成, 供下轮计算
    }

    /* ════════════════════════════════════════════════════════════
     * ══ Phase 3: Epilogue — 计算最后一个 tile ══
     * ════════════════════════════════════════════════════════════ */
    int last_buf = ((num_tiles - 1) & 1);
    compute_from(last_buf);

    /* ════ 写回 C ════ */
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = c_row_start + i;
        if (r >= M) break;

        if (c_col_start + TN - 1 < N) {
            #pragma unroll
            for (int j = 0; j < TN; j += F4) {
                float4 v;
                v.x = r_c[i][j + 0];
                v.y = r_c[i][j + 1];
                v.z = r_c[i][j + 2];
                v.w = r_c[i][j + 3];
                ST_FLOAT4(C[ROW_MAJOR(r, c_col_start + j, N)]) = v;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < TN; j++)
                if (c_col_start + j < N)
                    C[ROW_MAJOR(r, c_col_start + j, N)] = r_c[i][j];
        }
    }
}
