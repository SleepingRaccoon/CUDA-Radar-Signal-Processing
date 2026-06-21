/*
 * sgemm_tiled_v2_opt.cu — v2 优化版 (BM, BK, BN, TM, TN)
 *
 * v2 原始版本: sgemm_tiled_v2.cu
 *   每线程 TM×TN 个 C 元素, register tiling, blockDim = (BN/TN, BM/TM)
 *
 * 效率问题诊断:
 *   加载 sA (BM × BK):
 *     for (int kk = tx * TN; kk < BK; kk += BN)
 *     当 BK < BN 时, 只有 tx * TN < BK 的线程参与.
 *     例: BK=16, BN=64, TN=4 → TX=16, 仅 tx=0..3 (25%) 工作, 75% 闲置.
 *   加载 sB (BK × BN):
 *     for (int kk = ty * TM; kk < BK; kk += BM)
 *     同理, BK < BM 时大量线程闲置.
 *
 * 优化方案:
 *   ① flat tid 全线程协作加载, 每线程负责一个 float 元素
 *   ② 循环不变分支外提
 *
 * 注意: 本版本有 TM/TN register tiling, 但无 float4.
 *       仅修复加载阶段的线程利用率.
 *
 * 编译:
 *   nvcc -o sgemm_tiled_v2_opt sgemm_tiled_v2_opt.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

// ============================================================
// Kernel
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
__global__ void sgemm_tiled_v2_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int BLOCK_THREADS = TX * TY;

    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int tx  = threadIdx.x;   // [0, TX)
    int ty  = threadIdx.y;   // [0, TY)
    int tid = ty * TX + tx;

    // ── 共享内存 tile ──
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // ── 寄存器累积器 ──
    float rC[TM][TN] = {};

    // ── 循环不变的全局坐标 ──
    int base_row = by * BM + ty * TM;
    int base_col = bx * BN + tx * TN;

    for (int bk = 0; bk < K; bk += BK) {
        // ── 加载 sA: flat tid 全线程协作 ──
        for (int idx = tid; idx < BM * BK; idx += BLOCK_THREADS) {
            int r  = idx / BK;
            int c  = idx % BK;
            int gr = by * BM + r;
            int gc = bk + c;

            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // ── 加载 sB: flat tid 全线程协作 ──
        for (int idx = tid; idx < BK * BN; idx += BLOCK_THREADS) {
            int r  = idx / BN;
            int c  = idx % BN;
            int gr = bk + r;
            int gc = bx * BN + c;

            sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // ── 计算: register tiling ──
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int mi = 0; mi < TM; ++mi) {
                float av = sA[ty * TM + mi][kk];
                #pragma unroll
                for (int ni = 0; ni < TN; ++ni) {
                    rC[mi][ni] += av * sB[kk][tx * TN + ni];
                }
            }
        }

        __syncthreads();
    }

    // ── 写回 C ──
    for (int mi = 0; mi < TM; ++mi) {
        int r = base_row + mi;
        if (r >= M) break;

        int c0 = base_col;
        for (int ni = 0; ni < TN; ++ni) {
            int c = c0 + ni;
            if (c < N) C[r * N + c] = rC[mi][ni];
        }
    }
}


// ============================================================
// Launch 包装
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
void launch_sgemm_v2_opt(float* C, const float* A, const float* B,
                         int M, int N, int K, cudaStream_t stream = nullptr)
{
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    dim3 block_dim(TX, TY);
    dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tiled_v2_opt<BM, BK, BN, TM, TN>
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
    constexpr int BM = 64, BK = 16, BN = 64;
    constexpr int TM = 4,  TN = 4;
    constexpr int M = 997, N = 1003, K = 1019;

    printf("=== sgemm_tiled_v2_opt<BM=%d, BK=%d, BN=%d, TM=%d, TN=%d> ===\n",
           BM, BK, BN, TM, TN);
    printf("  M=%d N=%d K=%d (non-multiple)\n", M, N, K);
    printf("  blockDim=(%d,%d) gridDim=(%d,%d)  shared=%zuB\n",
           BN/TN, BM/TM, (N+BN-1)/BN, (M+BM-1)/BM,
           (BM*BK + BK*BN) * sizeof(float));
    printf("  Features: block tiling + register tiling (scalar mem, flat tid load)\n\n");

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
    launch_sgemm_v2_opt<BM, BK, BN, TM, TN>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    launch_sgemm_v2_opt<BM, BK, BN, TM, TN>(d_C, d_A, d_B, M, N, K);
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
