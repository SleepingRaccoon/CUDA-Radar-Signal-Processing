/*
 * sgemm_tiled_final.cu — 最终版通用矩阵乘法 C = A * B
 *
 * 设计演进:
 *   v1        — BM/BK/BN block tiling + shared memory, 1 element/thread
 *   v2        — +TM/TN register tiling (每线程 TM×TN 个元素)
 *   v3        — +FLOAT4 向量化访存
 *   user version — flat tid 全线程协作加载 shared memory（高利用率）
 *   ───────────────────────────────────────────────────────
 *   FINAL     — 融合上述所有优点
 *
 * loading 阶段: flat tid 映射, 所有线程参与, 每线程 1 个 float4
 * compute 阶段: (tx,ty) 2D 视角, 各读各的 shared memory
 * 边界处理    : +3 防越界 + 标量回退, 任意 M/N/K 均正确
 *
 * 约束:
 *   BM % TM == 0, BN % TN == 0
 *   BK % 4 == 0,  BN % 4 == 0,  TN % 4 == 0
 *
 * 编译:
 *   nvcc -o sgemm_tiled_final sgemm_tiled_final.cu
 */

#include <cuda_runtime.h>
#include "../include/macros.cuh"

#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

// ============================================================
// Kernel
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
__global__ void sgemm_tiled_final(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    // ── 编译期约束检查 ──
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(BK % 4 == 0,  "BK must be multiple of 4 for float4");
    static_assert(BN % 4 == 0,  "BN must be multiple of 4 for float4 (sB load)");
    static_assert(TN % 4 == 0,  "TN must be multiple of 4 for float4 (C write)");

    constexpr int TX = BN / TN;                     // block x 方向线程数
    constexpr int TY = BM / TM;                     // block y 方向线程数
    constexpr int BLOCK_THREADS = TX * TY;          // 本 block 总线程数

    int tx = threadIdx.x;                           // [0, TX)
    int ty = threadIdx.y;                           // [0, TY)
    int tid  = ty * TX + tx;                        // flat 线程索引
    int bx   = blockIdx.x;                          // C 的 tile 列
    int by   = blockIdx.y;                          // C 的 tile 行

    // ── 共享内存 tile ──
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // ── 寄存器累积器 ──
    float r_c[TM][TN] = {};

    // ── float4 辅助 ──
    constexpr int F4    = 4;
    constexpr int SA_F4 = BM * BK / F4;             // sA 的 float4 总数
    constexpr int SB_F4 = BK * BN / F4;             // sB 的 float4 总数

    // ============================================================
    // 主循环: 沿 K 维以 BK 为步长滑动
    // ============================================================
    for (int bk = 0; bk < K; bk += BK) {
        // ── 加载 s_a: BM × BK, flat tid 全线程协作 ──
        //    每个 float4 (= 4 floats) 由一个线程负责
        //    SA_F4 个 float4 → 用 stride = BLOCK_THREADS 循环分配
        for (int idx = tid; idx < SA_F4; idx += BLOCK_THREADS) {
            int row    = idx / (BK / F4);           // s_a 行: [0, BM)
            int col_st = (idx % (BK / F4)) * F4;    // s_a 列起点: 0, 4, 8, ...

            int gr     = by * BM + row;             // A 全局行号
            int gc     = bk + col_st;               // A 全局列号

            if (gr < M && gc + 3 < K) {
                // ── float4 主体路径 ──
                ST_FLOAT4(s_a[row][col_st]) = LD_FLOAT4(A[gr * K + gc]);
            } else if (gr < M && gc < K) {
                // ── 标量回退: K 尾部不足 4 个元素 ──
                #pragma unroll
                for (int t = 0; t < F4; ++t)
                    s_a[row][col_st + t] = (gc + t < K) ? A[gr * K + gc + t] : 0.0f;
            } else {
                // ── 底部越界行 (row >= M), 填充 0 ──
                #pragma unroll
                for (int t = 0; t < F4; ++t)
                    s_a[row][col_st + t] = 0.0f;
            }
        }

        // ── 加载 s_b: BK × BN, flat tid 全线程协作 ──
        for (int idx = tid; idx < SB_F4; idx += BLOCK_THREADS) {
            int row    = idx / (BN / F4);           // s_b 行: [0, BK)
            int col_st = (idx % (BN / F4)) * F4;    // s_b 列起点: 0, 4, 8, ...

            int gr     = bk + row;                  // B 全局行号
            int gc     = bx * BN + col_st;          // B 全局列号

            if (gr < K && gc + 3 < N) {
                // ── float4 主体路径 ──
                ST_FLOAT4(s_b[row][col_st]) = LD_FLOAT4(B[gr * N + gc]);
            } else if (gr < K && gc < N) {
                // ── 标量回退: N 尾部不足 4 个元素 ──
                #pragma unroll
                for (int t = 0; t < F4; ++t)
                    s_b[row][col_st + t] = (gc + t < N) ? B[gr * N + gc + t] : 0.0f;
            } else {
                // ── 行越界 (row >= K), 填充 0 ──
                #pragma unroll
                for (int t = 0; t < F4; ++t)
                    s_b[row][col_st + t] = 0.0f;
            }
        }

        __syncthreads();

        // ── 计算当前 K-tile 的局部乘积 ──
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int mi = 0; mi < TM; ++mi) {
                float av = s_a[ty * TM + mi][kk];
                #pragma unroll
                for (int ni = 0; ni < TN; ++ni) {
                    r_c[mi][ni] += av * s_b[kk][tx * TN + ni];
                }
            }
        }

        __syncthreads();
    }

    // ============================================================
    // 写回 C (float4 向量化)
    // ============================================================
    for (int mi = 0; mi < TM; ++mi) {
        int r = by * BM + ty * TM + mi;
        if (r >= M) break;          // 后续 mi 更大, 行号更大, 同样越界

        int c0 = bx * BN + tx * TN;

        if (c0 + TN - 1 < N) {
            // ── float4 主体路径: N 方向余量充足 ──
            #pragma unroll
            for (int ni = 0; ni < TN; ni += F4) {
                float4 v4;
                v4.x = r_c[mi][ni + 0];
                v4.y = r_c[mi][ni + 1];
                v4.z = r_c[mi][ni + 2];
                v4.w = r_c[mi][ni + 3];
                ST_FLOAT4(C[r * N + c0 + ni]) = v4;
            }
        } else {
            // ── 标量回退: 最右 tile, N 不足 TN ──
            #pragma unroll
            for (int ni = 0; ni < TN; ++ni) {
                int c = c0 + ni;
                if (c < N) C[r * N + c] = r_c[mi][ni];
            }
        }
    }
}


// ============================================================
// Host 端 launch 包装
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
void launch_sgemm_final(float* C, const float* A, const float* B,
                        int M, int N, int K,
                        cudaStream_t stream = nullptr)
{
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    dim3 block_dim(TX, TY);
    dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_tiled_final<BM, BK, BN, TM, TN>
        <<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
}


// ============================================================
// CPU 参考 (验证用)
// ============================================================
static void gemm_cpu(float* C, const float* A, const float* B,
                     int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}


// ============================================================
// 正确性检查
// ============================================================
static bool check_result(const float* C_gpu, const float* C_cpu,
                         int M, int N, float eps = 1e-3f)
{
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        float maxv = fmaxf(fabsf(C_gpu[i]), fabsf(C_cpu[i]));
        bool ok = (maxv > 1.0f) ? (diff / maxv <= eps) : (diff <= eps);
        if (!ok) {
            printf("  MISMATCH at [%d]: GPU=%f, CPU=%f, diff=%e\n",
                   i, C_gpu[i], C_cpu[i], diff);
            return false;
        }
    }
    return true;
}


// ============================================================
// RAII 计时器
// ============================================================
struct CudaTimer {
    cudaEvent_t start_, stop_;

    CudaTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~CudaTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }

    void begin() { cudaEventRecord(start_); }
    void end()   { cudaEventRecord(stop_); cudaEventSynchronize(stop_); }

    float ms() const {
        float t = 0;
        cudaEventElapsedTime(&t, start_, stop_);
        return t;
    }
};


// ============================================================
// Main
// ============================================================
int main()
{
    // ── 可配置参数 ──
    constexpr int BM = 64, BK = 16, BN = 64;
    constexpr int TM =  4, TN =  4;

    constexpr int M = 997;   // 故意用素数, 测试边界通用性
    constexpr int N = 1003;
    constexpr int K = 1019;

    // ── 打印配置 ──
    printf("sgemm_tiled_final<BM=%d, BK=%d, BN=%d, TM=%d, TN=%d>\n",
           BM, BK, BN, TM, TN);
    printf("  M=%d, N=%d, K=%d (non-multiple dimensions)\n", M, N, K);
    printf("  blockDim=(%d, %d), gridDim=(%d, %d)\n",
           BN / TN, BM / TM,
           (N + BN - 1) / BN, (M + BM - 1) / BM);
    printf("  shared mem = %zu bytes / block | float4: TN=%d, BK=%d\n",
           (BM * BK + BK * BN) * sizeof(float), TN, BK);

    // ── Host 内存 (C++ 风格) ──
    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    float* h_A     = new float[szA];
    float* h_B     = new float[szB];
    float* h_C_gpu = new float[szC];
    float* h_C_cpu = new float[szC];

    // ── 随机初始化 (C++ <random> 风格) ──
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < szA; ++i) h_A[i] = dist(rng);
    for (size_t i = 0; i < szB; ++i) h_B[i] = dist(rng);

    // ── Device 内存 ──
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szA * sizeof(float));
    cudaMalloc(&d_B, szB * sizeof(float));
    cudaMalloc(&d_C, szC * sizeof(float));

    cudaMemcpy(d_A, h_A, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, szB * sizeof(float), cudaMemcpyHostToDevice);

    // ── Warm-up ──
    launch_sgemm_final<BM, BK, BN, TM, TN>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    // ── 计时 ──
    CudaTimer timer;
    timer.begin();
    launch_sgemm_final<BM, BK, BN, TM, TN>(d_C, d_A, d_B, M, N, K);
    timer.end();

    cudaMemcpy(h_C_gpu, d_C, szC * sizeof(float), cudaMemcpyDeviceToHost);

    // ── CPU 验证 ──
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_C_cpu, h_A, h_B, M, N, K);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    // ── 正确性 ──
    bool ok = check_result(h_C_gpu, h_C_cpu, M, N);

    printf("\n── Results ──\n");
    printf("  GPU: %.3f ms  |  CPU: %.3f ms  |  Speedup: %.1f x\n",
           timer.ms(), cpu_ms, cpu_ms / timer.ms());
    printf("  %s\n", ok ? "✓ PASSED" : "✗ FAILED");

    // ── 清理 (C++ delete[]) ──
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return ok ? 0 : 1;
}
