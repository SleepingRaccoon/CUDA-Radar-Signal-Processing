#include <random>
#include <chrono>

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(func) do { \
    cudaError_t err = func; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR]: %s: %d: %s! \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

void random_init(float *a, int n) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < n; i++)
        a[i] = dis(gen);
}

int is_approx_equal(float *a, float *b, int n) {
    const float eps = 1e-3;
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps)
            return i;
    }
    return -1;
}

void sgemm(float *a, float *b, float *c, int M, int P, int N, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < P; k++) {
                sum += a[i * P + k] * b[k * N + j];
            }
            c[i * N + j] += alpha * sum + beta * c[i * N + j];
        }
    }
}

template <const int BLK>
__global__ void sgemm_v3(float *a, float *b, float *c, int M, int P, int N, float alpha, float beta) {
    
    const int BM = BLK;
    const int BP = BLK;
    const int BN = BLK;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int thread_id = threadIdx.x;

    int ty = thread_id / BN;
    int tx = thread_id % BN;

    int a_id_y = thread_id / BP;
    int a_id_x = thread_id % BP;

    int b_id_y = thread_id / BN;
    int b_id_x = thread_id % BN;

    __shared__ float sh_A[BM * BP];
    __shared__ float sh_B[BP * BN];

    float *A = &a[by * BM * P];
    float *B = &b[bx * BN];
    float *C = &c[by * BM * N + bx * BN];

    float sum = 0.0f;

    for (int k = 0; k < P; k += BP) {

        if (by * BM + a_id_y < M && k + a_id_x < P)
            sh_A[a_id_y * BP + a_id_x] = A[a_id_y * P + a_id_x];
        else 
            sh_A[a_id_y * BP + a_id_x] = 0.0f;
        
        if (k + b_id_y < P && bx * BN + b_id_x < N)
            sh_B[b_id_y * BN + b_id_x] = B[b_id_y * N + b_id_x];
        else 
            sh_B[b_id_y * BN + b_id_x] = 0.0f;

        __syncthreads();
        
        if (ty < BM) {
            for (int i = 0; i < BP; i++)
                sum += sh_A[ty * BP + i] * sh_B[i * BN + tx];
        }
        __syncthreads();

        A += BP;
        B += BP * N;
    }
    if (by * BM + ty < M && bx * BN + tx < N)
        C[ty * N + tx] = alpha * sum + beta * C[ty * N + tx];
}

int main() {

    const int M = 279;
    const int P = 534;
    const int N = 1029;

    const float al = 1.0f;
    const float be = 0.0f;

    const int sz_A = M * P;
    const int mem_A = sizeof(float) * sz_A;
    
    const int sz_B = P * N;
    const int mem_B = sizeof(float) * sz_B;

    const int sz_C = M * N;
    const int mem_C = sizeof(float) * sz_C;

    printf("\nMemory allocation at host...\n");
    float *h_A = new float [sz_A];
    float *h_B = new float [sz_B];
    float *h_C = new float [sz_C];
    float *h_C_ref = new float [sz_C];


    random_init(h_A, sz_A);
    random_init(h_B, sz_B);

    printf("\nCPU sgemm test...\n");
    auto h_t1 = std::chrono::high_resolution_clock::now();
    sgemm(h_A, h_B, h_C_ref, M, P, N, al, be);
    auto h_t2 = std::chrono::high_resolution_clock::now();
    float h_dt = std::chrono::duration<float, std::milli>(h_t2 - h_t1).count();
    printf("CPU sgemm time: %.4f ms. \n", h_dt);

    printf("\nMemory allocation at device...\n");
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_A, mem_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, mem_B));
    CUDA_CHECK(cudaMalloc((void **)&d_C, mem_C));

    printf("\nMemcpy from host to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mem_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mem_B, cudaMemcpyHostToDevice));

    cudaEvent_t d_t1, d_t2;
    CUDA_CHECK(cudaEventCreate(&d_t1));
    CUDA_CHECK(cudaEventCreate(&d_t2));
  
    const int BLK = 16;

    const int grid_y = CEIL_DIV(M, BLK);
    const int grid_x = CEIL_DIV(N, BLK); 

    dim3 block_sz(BLK * BLK);
    dim3 grid_sz(grid_x, grid_y);

    printf("\nGPU sgemm_v3 test...\n");
    CUDA_CHECK(cudaEventRecord(d_t1));
    sgemm_v3<BLK> <<<grid_sz, block_sz>>> (d_A, d_B, d_C, M, P, N, al, be);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(d_t2));
    CUDA_CHECK(cudaDeviceSynchronize());

    float d_dt = 0.0;
    CUDA_CHECK(cudaEventElapsedTime(&d_dt, d_t1, d_t2));
    printf("GPU sgemm_v3 time: %.4f ms. \n", d_dt);

    printf("\nSpeed up x%d\n", int(h_dt / d_dt));

    printf("\nMemcpy from device to host...\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, mem_C, cudaMemcpyDeviceToHost));

    int idx = is_approx_equal(h_C, h_C_ref, sz_C);
    if (idx == -1)
        printf("\nGPU sgemm_v3 test passed!\n");
    else 
        printf("\nMismatch at index (%d, %d)!\n", idx / N, idx % N);

    printf("\nClear resource...\n");

    CUDA_CHECK(cudaEventDestroy(d_t1));
    CUDA_CHECK(cudaEventDestroy(d_t2));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete [] h_A;
    delete [] h_B;
    delete [] h_C;
    delete [] h_C_ref;

    return 0;

}