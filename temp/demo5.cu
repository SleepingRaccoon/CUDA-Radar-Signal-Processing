/*
 * sgemm_tiled_v1_opt.cu — v1 优化版 (仅 BM, BK, BN)
 *
 * v1 原始版本: sgemm_tiled.cu
 *   每线程 1 个 C 元素, blockDim = (BN, BM)
 *
 * 效率问题诊断:
 *   加载 sA (BM × BK):
 *     for (int t = tx; t < BK; t += BN)
 *     当 BK < BN 时, 只有 tx < BK 的线程参与, tx >= BK 的线程闲置.
 *     例: BK=16, BN=64 → 仅 tx=0..15 (25%) 工作, 75% 闲置.
 *   加载 sB (BK × BN):
 *     for (int t = ty; t < BK; t += BM)
 *     同理, BK < BM 时大量线程闲置.
 *
 * 优化方案:
 *   ① flat tid 全线程协作加载, 每线程精确负责一个 float 元素
 *   ② 循环不变分支外提 (row<M, col<N 只需判断一次)
 *
 * 注意: 本版本不加 TM/TN register tiling, 不加 float4.
 *       每线程仍只算 1 个 C 元素, 仅修复加载阶段的线程利用率.
 *
 * 编译:
 *   nvcc -o sgemm_tiled_v1_opt sgemm_tiled_v1_opt.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

// ============================================================
// Kernel
// ============================================================
template <int BM, int BK, int BN>
__global__ void sgemm_tiled_v1_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;   // [0, BN)
    int ty = threadIdx.y;   // [0, BM)
    int tid = ty * blockDim.x + tx;

    int row = by * BM + ty;  // 循环不变
    int col = bx * BN + tx;  // 循环不变

    constexpr int BLOCK_THREADS = BM * BN;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float sum = 0.0f;

    bool row_ok = row < M;   // 外提
    bool col_ok = col < N;   // 外提

    for (int bk = 0; bk < K; bk += BK) {
        // ── 加载 sA: flat tid 全线程协作 ──
        //    sA 总共 BM×BK 个 float, BLOCK_THREADS 个线程平摊
        for (int idx = tid; idx < BM * BK; idx += BLOCK_THREADS) {
            int r  = idx / BK;             // sA 行: [0, BM)
            int c  = idx % BK;             // sA 列: [0, BK)
            int gr = by * BM + r;          // A 全局行
            int gc = bk + c;               // A 全局列

            if (gr < M && gc < K)
                sA[r][c] = A[gr * K + gc];
            else
                sA[r][c] = 0.0f;
        }

        // ── 加载 sB: flat tid 全线程协作 ──
        for (int idx = tid; idx < BK * BN; idx += BLOCK_THREADS) {
            int r  = idx / BN;             // sB 行: [0, BK)
            int c  = idx % BN;             // sB 列: [0, BN)
            int gr = bk + r;               // B 全局行
            int gc = bx * BN + c;          // B 全局列

            if (gr < K && gc < N)
                sB[r][c] = B[gr * N + gc];
            else
                sB[r][c] = 0.0f;
        }

        __syncthreads();

        // ── 计算 ──
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            sum += sA[ty][kk] * sB[kk][tx];
        }

        __syncthreads();
    }

    // ── 写回 C (循环不变分支已外提) ──
    if (row_ok && col_ok) {
        C[row * N + col] = sum;
    }
}


// ============================================================
// Launch 包装
// ============================================================
template <int BM, int BK, int BN>
void launch_sgemm_v1_opt(float* C, const float* A, const float* B,
                         int M, int N, int K, cudaStream_t stream = nullptr)
{
    dim3 block_dim(BN, BM);
    dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tiled_v1_opt<BM, BK, BN>
        <<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
}


// ============================================================
// CPU 参考
// ============================================================
static void gemm_cpu(float* C, const float* A, const float* B,
                     int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}


// ============================================================
// 验证
// ============================================================
static bool check_result(const float* C_gpu, const float* C_cpu,
                         int M, int N, float eps = 1e-3f)
{
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        float maxv = fmaxf(fabsf(C_gpu[i]), fabsf(C_cpu[i]));
        bool ok = (maxv > 1.0f) ? (diff / maxv <= eps) : (diff <= eps);
        if (!ok) {
            printf("  MISMATCH [%d]: GPU=%f CPU=%f diff=%e\n",
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
    float ms() const { float t; cudaEventElapsedTime(&t, start_, stop_); return t; }
};


// ============================================================
// main
// ============================================================
int main()
{
    // ── 统一参数 ──
    constexpr int BM = 64, BK = 16, BN = 64;
    constexpr int M = 997, N = 1003, K = 1019;

    printf("=== sgemm_tiled_v1_opt<BM=%d, BK=%d, BN=%d> ===\n", BM, BK, BN);
    printf("  M=%d N=%d K=%d (non-multiple)\n", M, N, K);
    printf("  blockDim=(%d,%d) gridDim=(%d,%d)  shared=%zuB\n",
           BN, BM, (N+BN-1)/BN, (M+BM-1)/BM,
           (BM*BK + BK*BN) * sizeof(float));
    printf("  Features: block tiling only (1 elt/thread, scalar mem, flat tid load)\n\n");

    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    float* h_A  = new float[szA];
    float* h_B  = new float[szB];
    float* h_Cg = new float[szC];
    float* h_Cc = new float[szC];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < szA; ++i) h_A[i] = dist(rng);
    for (size_t i = 0; i < szB; ++i) h_B[i] = dist(rng);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szA * sizeof(float));
    cudaMalloc(&d_B, szB * sizeof(float));
    cudaMalloc(&d_C, szC * sizeof(float));
    cudaMemcpy(d_A, h_A, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, szB * sizeof(float), cudaMemcpyHostToDevice);

    // warm-up
    launch_sgemm_v1_opt<BM, BK, BN>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    launch_sgemm_v1_opt<BM, BK, BN>(d_C, d_A, d_B, M, N, K);
    timer.end();
    cudaMemcpy(h_Cg, d_C, szC * sizeof(float), cudaMemcpyDeviceToHost);

    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_Cc, h_A, h_B, M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = check_result(h_Cg, h_Cc, M, N);
    printf("\n── Results ──\n");
    printf("  GPU: %.3f ms  CPU: %.3f ms  Speedup: %.1fx\n",
           timer.ms(), cpu_ms, cpu_ms / timer.ms());
    printf("  %s\n", ok ? "✓ PASSED" : "✗ FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_Cg; delete[] h_Cc;
    return ok ? 0 : 1;
}
