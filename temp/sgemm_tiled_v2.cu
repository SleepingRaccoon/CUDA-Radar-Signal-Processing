/*
 * sgemm_tiled_v2.cu — 通用矩阵乘法 C = A * B
 *
 * 相比 v1（1 element/thread），引入了寄存器分块（register tiling）：
 *   每个线程负责计算 TM × TN 个 C 元素，提高计算密度和寄存器复用率。
 *
 * 分块策略（三层分块）:
 *   1. Block tiling:   grid 将 C 分为 (BM × BN) 的 tile
 *   2. Shared memory:  K 维按 BK 分块，每次加载到 shared memory 减少全局访存
 *   3. Register tiling: 每个线程计算 TM × TN 子块，数据累积在寄存器
 *
 * 模板参数:
 *   BM — C 的 block tile 行数
 *   BK — K 维分块大小
 *   BN — C 的 block tile 列数
 *   TM — 每个线程在行方向负责的元素数
 *   TN — 每个线程在列方向负责的元素数
 *
 * 约束:
 *   BM % TM == 0, BN % TN == 0（建议，否则边缘线程需处理边界）
 *   (BM / TM) * (BN / TN) ≤ 1024（block 线程数上限）
 *   (BM * BK + BK * BN) * sizeof(float) ≤ 48KB（shared memory 上限）
 *
 * 编译:
 *   nvcc -o sgemm_tiled_v2 sgemm_tiled_v2.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdlib>

// ============================================================
// Kernel（C++ 风格模板参数）
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
__global__ void sgemm_tiled_v2(float* __restrict__ C,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                int M, int N, int K) {
    // --- 线程/block 索引 ---
    constexpr int TX = BN / TN;           // block x 方向线程数
    constexpr int TY = BM / TM;           // block y 方向线程数

    int bx = blockIdx.x;                  // C 的 tile 列索引
    int by = blockIdx.y;                  // C 的 tile 行索引
    int tx = threadIdx.x;                 // 线程在 block 中的 x（[0, TX)）
    int ty = threadIdx.y;                 // 线程在 block 中的 y（[0, TY)）

    // --- shared memory tile ---
    __shared__ float sA[BM][BK];          // A 的子块: BM × BK
    __shared__ float sB[BK][BN];          // B 的子块: BK × BN

    // --- 寄存器累积 ---
    float sum[TM][TN] = {};               // 零初始化（C++ 语法）

    // ============================================================
    // 主循环：沿 K 维滑动
    // ============================================================
    for (int k = 0; k < K; k += BK) {
        // ---- 1. 加载 sA (BM × BK) ----
        // 每个线程负责 TM 行，每行从 tx*TN 开始，按 BN 步长覆盖 BK
        for (int i = 0; i < TM; ++i) {
            int local_row = ty * TM + i;          // sA 行号: [0, BM)
            int global_row = by * BM + local_row; // A 全局行号

            for (int kk = tx * TN; kk < BK; kk += TX * TN) {
                // 一次加载 TN 个连续元素
                for (int j = 0; j < TN; ++j) {
                    int c = kk + j;               // sA 列号: [0, BK)
                    bool row_ok = global_row < M;
                    bool col_ok = (k + c) < K;
                    sA[local_row][c] = (row_ok && col_ok)
                                       ? A[global_row * K + k + c]
                                       : 0.0f;
                }
            }
        }

        // ---- 2. 加载 sB (BK × BN) ----
        // 每个线程负责 TN 列，每列从 ty*TM 开始，按 BM 步长覆盖 BK
        for (int j = 0; j < TN; ++j) {
            int local_col = tx * TN + j;          // sB 列号: [0, BN)
            int global_col = bx * BN + local_col; // B 全局列号

            for (int kk = ty * TM; kk < BK; kk += TY * TM) {
                // 一次加载 TM 个连续元素
                for (int i = 0; i < TM; ++i) {
                    int r = kk + i;               // sB 行号: [0, BK)
                    bool row_ok = (k + r) < K;
                    bool col_ok = global_col < N;
                    sB[r][local_col] = (row_ok && col_ok)
                                       ? B[(k + r) * N + global_col]
                                       : 0.0f;
                }
            }
        }

        __syncthreads();

        // ---- 3. 计算当前 K-tile 的局部乘积 ----
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int ti = 0; ti < TM; ++ti) {
                float a_val = sA[ty * TM + ti][kk];  // 可复用的 A 值
                #pragma unroll
                for (int tj = 0; tj < TN; ++tj) {
                    sum[ti][tj] += a_val * sB[kk][tx * TN + tj];
                }
            }
        }

        __syncthreads();
    }

    // ============================================================
    // 4. 写回全局内存
    // ============================================================
    for (int ti = 0; ti < TM; ++ti) {
        for (int tj = 0; tj < TN; ++tj) {
            int r = by * BM + ty * TM + ti;
            int c = bx * BN + tx * TN + tj;
            if (r < M && c < N) {
                C[r * N + c] = sum[ti][tj];
            }
        }
    }
}


// ============================================================
// Host 端 launch 包装（C++ 风格，默认 stream = 0）
// ============================================================
template <int BM, int BK, int BN, int TM, int TN>
void launch_sgemm_tiled_v2(float* C, const float* A, const float* B,
                            int M, int N, int K,
                            cudaStream_t stream = nullptr) {
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    dim3 block_dim(TX, TY);
    dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_tiled_v2<BM, BK, BN, TM, TN>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}


// ============================================================
// 验证 — CPU 参考
// ============================================================
void gemm_cpu(float* C, const float* A, const float* B,
              int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


// ============================================================
// 正确性检查
// ============================================================
bool check_result(const float* C_gpu, const float* C_cpu,
                  int M, int N, float eps = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        float max_val = fmaxf(fabsf(C_gpu[i]), fabsf(C_cpu[i]));
        bool ok = (max_val > 1.0f) ? (diff / max_val <= eps) : (diff <= eps);
        if (!ok) {
            printf("  MISMATCH at [%d]: GPU=%f, CPU=%f, diff=%f\n",
                   i, C_gpu[i], C_cpu[i], diff);
            return false;
        }
    }
    return true;
}


// ============================================================
// 简易计时器（RAII 风格）
// ============================================================
struct CudaTimer {
    cudaEvent_t start_, stop_;

    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void begin() { cudaEventRecord(start_); }
    void end()   { cudaEventRecord(stop_); cudaEventSynchronize(stop_); }
    float ms() {
        float t = 0;
        cudaEventElapsedTime(&t, start_, stop_);
        return t;
    }
};


// ============================================================
// Main 测试
// ============================================================
int main() {
    // ==================== 可配置参数 ====================
    constexpr int BM = 64, BK = 16, BN = 64;
    constexpr int TM =  4, TN =  4;

    int M = 256;
    int N = 256;
    int K = 256;
    // ===================================================

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    // --- Host 内存 ---
    float* h_A    = (float*)malloc(szA);
    float* h_B    = (float*)malloc(szB);
    float* h_C_gpu= (float*)malloc(szC);
    float* h_C_cpu= (float*)malloc(szC);

    // 随机初始化
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

    // --- 启动 kernel ---
    printf("sgemm_tiled_v2<BM=%d, BK=%d, BN=%d, TM=%d, TN=%d>\n",
           BM, BK, BN, TM, TN);
    printf("  M=%d, N=%d, K=%d\n", M, N, K);
    printf("  gridDim=(%d, %d), blockDim=(%d, %d)\n",
           (N + BN - 1) / BN, (M + BM - 1) / BM, BN / TN, BM / TM);
    printf("  Shared memory: %zu bytes / thread-block\n",
           (BM * BK + BK * BN) * sizeof(float));

    CudaTimer timer;
    timer.begin();
    launch_sgemm_tiled_v2<BM, BK, BN, TM, TN>(d_C, d_A, d_B, M, N, K);
    timer.end();

    cudaMemcpy(h_C_gpu, d_C, szC, cudaMemcpyDeviceToHost);

    // --- CPU 验证 ---
    gemm_cpu(h_C_cpu, h_A, h_B, M, N, K);

    bool ok = check_result(h_C_gpu, h_C_cpu, M, N);
    printf("%s (%.3f ms)\n", ok ? "✓ PASSED" : "✗ FAILED", timer.ms());

    // --- 资源释放 ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return ok ? 0 : 1;
}
