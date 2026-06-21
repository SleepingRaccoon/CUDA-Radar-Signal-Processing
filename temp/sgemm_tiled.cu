/*
 * sgemm_tiled.cu — 通用矩阵乘法 C = A * B
 *
 * 思想：将输出矩阵 C 划分为 BM×BN 的 tile，
 *       每个 block 负责计算一个 tile。
 *       K 维分块大小为 BK，将 A/B 的子块加载到 shared memory，
 *       减少全局内存访问次数。
 *
 * 参数：
 *   BM — C 矩阵每个 tile 的行数（也是 block 的 y 方向线程数）
 *   BK — K 维的分块大小
 *   BN — C 矩阵每个 tile 的列数（也是 block 的 x 方向线程数）
 *
 * 线程配置：
 *   blockDim  = (BN, BM)   — 每个线程计算 C 中的一个元素
 *   gridDim   = (ceil(N/BN), ceil(M/BM))
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// ============================================================
// Kernel
// ============================================================
template <int BM, int BK, int BN>
__global__ void sgemm_tiled(float* C, const float* A, const float* B,
                            int M, int N, int K) {
    // Block 在 grid 中的索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread 在 block 中的索引
    int tx = threadIdx.x;   // [0, BN)
    int ty = threadIdx.y;   // [0, BM)

    // 当前线程负责的 C 元素的行号和列号
    int row = by * BM + ty;
    int col = bx * BN + tx;

    // Shared memory 中的 tile
    __shared__ float sA[BM][BK];   // A 的子块: BM × BK
    __shared__ float sB[BK][BN];   // B 的子块: BK × BN

    float sum = 0.0f;

    // 沿着 K 维以步长 BK 滑动
    for (int k = 0; k < K; k += BK) {
        // ---- 加载 sA ----
        // sA[ty][0..BK) 需要 BK 个元素，但 x 方向只有 BN 个线程
        // 当 BK > BN 时，每个线程需要加载多个元素
        if (row < M) {
            for (int t = tx; t < BK; t += BN) {
                int k_idx = k + t;
                sA[ty][t] = (k_idx < K) ? A[row * K + k_idx] : 0.0f;
            }
        } else {
            // row 越界，填充 0
            for (int t = tx; t < BK; t += BN) {
                sA[ty][t] = 0.0f;
            }
        }

        // ---- 加载 sB ----
        // sB[0..BK)[tx] 需要 BK 个元素，但 y 方向只有 BM 个线程
        // 当 BK > BM 时，每个线程需要加载多个元素
        if (col < N) {
            for (int t = ty; t < BK; t += BM) {
                int k_idx = k + t;
                sB[t][tx] = (k_idx < K) ? B[k_idx * N + col] : 0.0f;
            }
        } else {
            for (int t = ty; t < BK; t += BM) {
                sB[t][tx] = 0.0f;
            }
        }

        __syncthreads();

        // ---- 计算当前子块的乘积 ----
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    // ---- 写回全局内存 ----
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// ============================================================
// Host 端包装函数
// ============================================================
template <int BM, int BK, int BN>
void launch_sgemm_tiled(float* d_C, const float* d_A, const float* d_B,
                        int M, int N, int K, cudaStream_t stream = 0) {
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_tiled<BM, BK, BN><<<gridDim, blockDim, 0, stream>>>(d_C, d_A, d_B, M, N, K);
}


// ============================================================
// 验证 — CPU 参考实现
// ============================================================
void gemm_cpu(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================
// 正确性检查
// ============================================================
bool check_result(const float* C_gpu, const float* C_cpu, int M, int N, float eps = 1e-3f) {
    int total = M * N;
    for (int i = 0; i < total; i++) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        float max_val = fmaxf(fabsf(C_gpu[i]), fabsf(C_cpu[i]));
        if (max_val > 1.0f && diff / max_val > eps) {
            printf("  MISMATCH at [%d]: GPU=%f, CPU=%f, rel_err=%f\n",
                   i, C_gpu[i], C_cpu[i], diff / max_val);
            return false;
        }
        if (max_val <= 1.0f && diff > eps) {
            printf("  MISMATCH at [%d]: GPU=%f, CPU=%f, abs_err=%f\n",
                   i, C_gpu[i], C_cpu[i], diff);
            return false;
        }
    }
    return true;
}


// ============================================================
// Main 测试
// ============================================================
int main() {
    // 测试参数：可自行修改 BM, BK, BN 和矩阵尺寸
    constexpr int BM = 16;
    constexpr int BK = 16;
    constexpr int BN = 16;

    int M = 128;
    int N = 128;
    int K = 128;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // 分配 host 内存
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_gpu = (float*)malloc(size_C);
    float* h_C_cpu = (float*)malloc(size_C);

    // 随机初始化 A 和 B
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 10.0f;

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 启动 kernel
    printf("Launching sgemm_tiled<BM=%d, BK=%d, BN=%d>\n", BM, BK, BN);
    printf("  M=%d, N=%d, K=%d\n", M, N, K);
    printf("  gridDim=(%d, %d), blockDim=(%d, %d)\n",
           (N + BN - 1) / BN, (M + BM - 1) / BM, BN, BM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_sgemm_tiled<BM, BK, BN>(d_C, d_A, d_B, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 取回结果
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // CPU 参考
    gemm_cpu(h_C_cpu, h_A, h_B, M, N, K);

    // 验证
    bool ok = check_result(h_C_gpu, h_C_cpu, M, N);
    if (ok) {
        printf("✓ PASSED (%.3f ms)\n", ms);
    } else {
        printf("✗ FAILED\n");
    }

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ok ? 0 : 1;
}
