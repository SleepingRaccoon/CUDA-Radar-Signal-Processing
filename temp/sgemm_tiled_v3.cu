/*
 * sgemm_tiled_v3.cu — 通用矩阵乘法 C = A * B
 *
 * 在 v2（register tiling: TM×TN/thread）基础上，引入 float4 向量化访存优化。
 *
 * float4 优化的核心收益：
 *   一次 128-bit 加载 = 4 个 float，将全局内存访问次数减少为 1/4，
 *   更好地利用显存带宽（GDDR 对连续大请求的带宽利用率更高）。
 *
 * 应用位置（三处全局内存访问）:
 *   1. 加载 A 到 shared memory — 行主序，TN 个连续元素 → float4
 *   2. 加载 B 到 shared memory — 行主序，每行 TN 个连续元素 → float4
 *   3. 写回 C 到全局内存   — 行主序，TN 个连续元素 → float4
 *
 * 约束:
 *   TN % 4 == 0      （否则退化为标量加载）
 *   K  % 4 == 0      （保证地址 16 字节对齐，建议值）
 *   LD_FLOAT4 / ST_FLOAT4 定义见 include/macros.cuh
 *
 * 编译:
 *   nvcc -o sgemm_tiled_v3 sgemm_tiled_v3.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdlib>

#include "../include/macros.cuh"

// ============================================================
// Kernel
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
__global__ void sgemm_tiled_v3(float* __restrict__ C,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                int M, int N, int K) {
    // --- block/thread 索引 ---
    constexpr int TX = BN / TN;           // block x 方向线程数
    constexpr int TY = BM / TM;           // block y 方向线程数

    int bx = blockIdx.x;                  // C tile 列索引
    int by = blockIdx.y;                  // C tile 行索引
    int tx = threadIdx.x;                 // [0, TX)
    int ty = threadIdx.y;                 // [0, TY)

    // --- 共享内存 tile ---
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // --- 寄存器累积器 ---
    float sum[TM][TN] = {};

    // --- float4 辅助常量 ---
    constexpr int VEC4 = 4;                     // float4 = 4 floats
    constexpr int VEC_N = TN / VEC4;            // 每 segment 的 float4 次数

    // ============================================================
    // 主循环：沿 K 维分块滑动
    // ============================================================
    for (int k = 0; k < K; k += BK) {
        // ——————————————————————————————————————————
        // 1. 加载 sA: BM × BK
        //    每个线程加载 TM 行，每行在 [tx*TN, BK) 范围内
        //    步长为 TX*TN = BN，每步通过 float4 加载 TN 个元素
        // ——————————————————————————————————————————
        for (int i = 0; i < TM; ++i) {
            int local_row = ty * TM + i;
            int global_row = by * BM + local_row;

            for (int kk = tx * TN; kk < BK; kk += TX * TN) {
                int col_base = k + kk;          // 全局 K 坐标起点
                bool row_ok  = global_row < M;

                // ---- 主体路径：float4 向量化加载 ----
                // 条件：行不越界，且 TN 个连续列均在 K 范围内
                if (row_ok && (col_base + TN - 1 < K)) {
                    #pragma unroll
                    for (int v = 0; v < VEC_N; ++v) {
                        int off = v * VEC4;
                        ST_FLOAT4(sA[local_row][kk + off]) =
                            LD_FLOAT4(A[global_row * K + col_base + off]);
                    }
                }
                // ---- 边界路径：标量回退 ----
                else {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        int c    = kk + j;
                        int gc   = col_base + j;
                        sA[local_row][c] = (row_ok && gc < K)
                                           ? A[global_row * K + gc] : 0.0f;
                    }
                }
            }
        }

        // ——————————————————————————————————————————
        // 2. 加载 sB: BK × BN
        //    每个线程覆盖 BK 中的 TM 行（步长 BM），
        //    每行在 BN 中取 TN 个连续元素 → float4 向量化
        // ——————————————————————————————————————————
        for (int kk = ty * TM; kk < BK; kk += TY * TM) {
            for (int i = 0; i < TM; ++i) {
                int r        = kk + i;                       // sB 行号
                int row_idx  = k + r;                        // B 全局行
                int col_base = bx * BN + tx * TN;            // B 全局列起点
                bool row_ok  = row_idx < K;

                // ---- float4 路径 ----
                if (row_ok && (col_base + TN - 1 < N)) {
                    #pragma unroll
                    for (int v = 0; v < VEC_N; ++v) {
                        int off = v * VEC4;
                        ST_FLOAT4(sB[r][tx * TN + off]) =
                            LD_FLOAT4(B[row_idx * N + col_base + off]);
                    }
                }
                // ---- 标量回退 ----
                else {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        int gc = col_base + j;
                        sB[r][tx * TN + j] = (row_ok && gc < N)
                                             ? B[row_idx * N + gc] : 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        // ——————————————————————————————————————————
        // 3. 计算当前 K-tile 的局部乘积
        //    sA[ty*TM+ti][kk] 一次读取，对 TN 个 sB 元素复用
        // ——————————————————————————————————————————
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int ti = 0; ti < TM; ++ti) {
                float a_val = sA[ty * TM + ti][kk];
                #pragma unroll
                for (int tj = 0; tj < TN; ++tj) {
                    sum[ti][tj] += a_val * sB[kk][tx * TN + tj];
                }
            }
        }

        __syncthreads();
    }

    // ——————————————————————————————————————————
    // 4. 将结果写回全局内存（float4 向量化）
    // ——————————————————————————————————————————
    for (int ti = 0; ti < TM; ++ti) {
        int r        = by * BM + ty * TM + ti;
        int col_base = bx * BN + tx * TN;
        bool row_ok  = r < M;

        // ---- float4 路径 ----
        if (row_ok && (col_base + TN - 1 < N)) {
            #pragma unroll
            for (int v = 0; v < VEC_N; ++v) {
                int off = v * VEC4;
                float4 v4;
                v4.x = sum[ti][off + 0];
                v4.y = sum[ti][off + 1];
                v4.z = sum[ti][off + 2];
                v4.w = sum[ti][off + 3];
                ST_FLOAT4(C[r * N + col_base + off]) = v4;
            }
        }
        // ---- 标量回退 ----
        else if (row_ok) {
            #pragma unroll
            for (int tj = 0; tj < TN; ++tj) {
                int gc = col_base + tj;
                if (gc < N) C[r * N + gc] = sum[ti][tj];
            }
        }
    }
}


// ============================================================
// Host 端 launch 包装
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
void launch_sgemm_tiled_v3(float* C, const float* A, const float* B,
                            int M, int N, int K,
                            cudaStream_t stream = nullptr) {
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    dim3 block_dim(TX, TY);
    dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_tiled_v3<BM, BK, BN, TM, TN>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}


// ============================================================
// CPU 参考
// ============================================================
void gemm_cpu(float* C, const float* A, const float* B,
              int M, int N, int K) {
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
                          int M, int N, float eps = 1e-3f) {
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
int main() {
    // ==================== 可配置参数 ====================
    constexpr int BM = 64, BK = 16, BN = 64;
    constexpr int TM =  4, TN =  4;           // TN % 4 == 0 以启用 float4

    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    // ===================================================

    // --- 编译时校验 ---
    static_assert(TN % 4 == 0, "TN must be a multiple of 4 for float4 optimization");

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    // --- Host 内存 ---
    float* h_A     = (float*)malloc(szA);
    float* h_B     = (float*)malloc(szB);
    float* h_C_gpu = (float*)malloc(szC);
    float* h_C_cpu = (float*)malloc(szC);

    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(rand() % 100) / 10.0f;

    // --- Device 内存 ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szA);
    cudaMalloc(&d_B, szB);
    cudaMalloc(&d_C, szC);

    cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice);

    // --- 启动 ---
    printf("sgemm_tiled_v3<BM=%d, BK=%d, BN=%d, TM=%d, TN=%d>\n",
           BM, BK, BN, TM, TN);
    printf("  M=%d, N=%d, K=%d\n", M, N, K);
    printf("  blockDim=(%d, %d), gridDim=(%d, %d)\n",
           BN / TN, BM / TM,
           (N + BN - 1) / BN, (M + BM - 1) / BM);
    printf("  shared mem = %zu bytes  |  float4 %s\n",
           (BM * BK + BK * BN) * sizeof(float),
           (TN >= 4 && K % 4 == 0) ? "ENABLED" : "disabled (TN<4 or K%4!=0)");

    CudaTimer timer;
    timer.begin();
    launch_sgemm_tiled_v3<BM, BK, BN, TM, TN>(d_C, d_A, d_B, M, N, K);
    timer.end();

    cudaMemcpy(h_C_gpu, d_C, szC, cudaMemcpyDeviceToHost);

    // --- CPU 验证 ---
    gemm_cpu(h_C_cpu, h_A, h_B, M, N, K);

    bool ok = check_result(h_C_gpu, h_C_cpu, M, N);
    printf("%s  (%.3f ms)\n", ok ? "✓ PASSED" : "✗ FAILED", timer.ms());

    // --- 清理 ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return ok ? 0 : 1;
}
