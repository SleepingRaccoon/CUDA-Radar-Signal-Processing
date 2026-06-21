/*
 * sgemm_tiled.cu  -- tiled GEMM with clear index mapping.
 *
 * Parameters (all compile-time constants via template):
 *   BM = rows of A tile loaded into shared memory per iteration
 *   BK = cols of A tile (also rows of B tile)
 *   BN = cols of B tile
 *   TM = rows of C computed by one thread
 *   TN = cols of C computed by one thread
 *
 * Block: (BN/TN, BM/TM) threads -- each thread computes TM x TN = TM*TN elements
 *
 * Tile loading layout (256 threads with float4):
 *   s_a[BM][BK]:  BM/BK_THREADS rows x BK/4 cols  (BK_THREADS = BK/4)
 *                 thread id: row = tid / BK_THREADS, col4 = tid % BK_THREADS
 *   s_b[BK][BN]:  BK/BN_THREADS rows x BN/4 cols   (BN_THREADS = BN/4)
 *                 thread id: row = tid / BN_THREADS, col4 = tid % BN_THREADS
 *
 *   float4 loads 4 floats = BK/BN column stride per thread.
 *   Total threads = BM/ (BK/4)  for s_a = BM*BK/4 / 4 ... wait let me recalc.
 *
 *   s_a fills BM*BK floats. With N_THREADS = block.x * block.y threads,
 *   each thread handles 4 floats (float4): need BM*BK/4 threads.
 *   We must ensure N_THREADS >= BM*BK/4 (pad with inactive threads if needed).
 *
 *   Similarly s_b fills BK*BN floats, need BK*BN/4 threads.
 */

#include <cuda_runtime.h>

#define OFFSET(row, col, ld)  ((row) * (ld) + (col))
#define FLOAT4(ptr)           (*reinterpret_cast<float4 *>(&(ptr)))

// ---------------------------------------------------------------------------
// Tiled GEMM
// ---------------------------------------------------------------------------
//  BM, BN  = C tile size per block
//  BK      = inner tile size (K dimension)
//  TM, TN  = C sub-tile size per thread
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K)
{
    // This thread's position in the block
    const int tx = threadIdx.x;  // col index within block (0..BN/TN-1)
    const int ty = threadIdx.y;  // row index within block (0..BM/TM-1)
    const int tid = ty * blockDim.x + tx;

    // This block's position in the grid
    const int bx = blockIdx.x;   // col block index (0..ceil(N/BN)-1)
    const int by = blockIdx.y;   // row block index (0..ceil(M/BM)-1)

    // Shared memory tiles
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // Register accumulator for this thread's TM x TN C-elements
    float r_c[TM][TN] = {0.0f};

    // ---------------------------------------------------------------
    // LOAD s_a: each thread loads 4 floats from A into shared memory
    //   s_a layout: BM rows x BK cols
    //   we have BM*BK floats to load, each thread loads 4 floats
    //   -> need BM*BK/4 threads
    //   BM*BK/4 = 128*8/4 = 256  (when BM=128, BK=8)
    //   which happens to equal the block size (BN/TN)*(BM/TM)
    //   = (128/8)*(128/8) = 16*16 = 256
    //
    //   Index mapping:
    //     BM / (BK/4) = 128 / 2 = 64 "groups" of threads, each group
    //     handles 2 float4-lanes = BK cols.  But 64 * 2 = 128, not 256.
    //     Actually:  BK/4 = 2, so each row of s_a needs 2 threads.
    //     BM * (BK/4) = 128 * 2 = 256 threads total.
    //
    //     row of s_a = tid / (BK/4)     = tid / 2    (0..127)
    //     col group   = tid % (BK/4)     = tid & 1    (0 or 1)
    //     actual col  = col_group * 4   = (tid%2)*4  (0 or 4)
    // ---------------------------------------------------------------
    constexpr int SA_COL_GROUPS = BK / 4;            // how many float4-groups per row
    int sa_row = tid / SA_COL_GROUPS;                 // 0..BM-1
    int sa_col_start = (tid % SA_COL_GROUPS) * 4;    // 0, 4, ..., BK-4

    // ---------------------------------------------------------------
    // LOAD s_b: each thread loads 4 floats from B into shared memory
    //   s_b layout: BK rows x BN cols
    //   total floats = BK*BN, at 4 per thread = BK*BN/4 threads
    //   With BK=8, BN=128: 8*128/4 = 256 threads ✓
    //
    //     row of s_b = tid / (BN/4)     = tid / 32   (0..7)
    //     col group   = tid % (BN/4)     = tid & 31   (0..31)
    //     actual col  = col_group * 4   = (tid%32)*4 (0,4,...,124)
    // ---------------------------------------------------------------
    constexpr int SB_COL_GROUPS = BN / 4;             // how many float4-groups per row
    int sb_row = tid / SB_COL_GROUPS;                 // 0..BK-1
    int sb_col_start = (tid % SB_COL_GROUPS) * 4;     // 0, 4, ..., BN-4

    // ---------------------------------------------------------------
    // Main loop over K in chunks of BK
    // ---------------------------------------------------------------
    for (int bk = 0; bk < K; bk += BK) {

        // ---- load A tile from global to shared ----
        if (sa_row < BM && bk + sa_col_start < K) {
            int gmem_row = by * BM + sa_row;
            int gmem_col = bk + sa_col_start;
            FLOAT4(s_a[sa_row][sa_col_start]) = FLOAT4(A[OFFSET(gmem_row, gmem_col, K)]);
        }

        // ---- load B tile from global to shared ----
        if (sb_row < BK && bx * BN + sb_col_start + 3 < N && bk + sb_row < K) {
            int gmem_row = bk + sb_row;
            int gmem_col = bx * BN + sb_col_start;
            FLOAT4(s_b[sb_row][sb_col_start]) = FLOAT4(B[OFFSET(gmem_row, gmem_col, N)]);
        }

        __syncthreads();

        // ---- compute: TM x TN partial products ----
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                float a_val = s_a[ty * TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += a_val * s_b[k][tx * TN + n];
                }
            }
        }

        __syncthreads();
    }

    // ---------------------------------------------------------------
    // Store results: this thread's TM x TN tile of C
    //   Global row = by*BM + ty*TM + i    (i = 0..TM-1)
    //   Global col = bx*BN + tx*TN + j    (j = 0..TN-1)
    // ---------------------------------------------------------------
    for (int i = 0; i < TM; i++) {
        int gmem_row = by * BM + ty * TM + i;
        if (gmem_row >= M) continue;
        for (int j = 0; j < TN; j += 4) {
            int gmem_col = bx * BN + tx * TN + j;
            if (gmem_col + 3 < N) {
                FLOAT4(C[OFFSET(gmem_row, gmem_col, N)]) = FLOAT4(r_c[i][j]);
            } else {
                // tail handling for last column out-of-range
                for (int t = 0; t < 4 && gmem_col + t < N; t++)
                    C[OFFSET(gmem_row, gmem_col + t, N)] = r_c[i][j + t];
            }
        }
    }
}
