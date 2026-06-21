/* ===========================================================================
 * reduce_shfl FIXED
 *
 * Bug: __shfl_down_sync must be called by ALL threads in the warp,
 *      but the original wrapped it in if(tid < offset), which made
 *      threads with tid >= offset skip the call — UB.
 *
 * Fix: pull shuffle outside the if, only use the result conditionally.
 * ===========================================================================
template <typename BLOCK_SZ>
__global__ void reduce_shfl_fixed(float *d_x, int n, float *d_y) {
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid  = threadIdx.x;
    __shared__ float sh_mem[BLOCK_SZ];
    sh_mem[tid] = (g_tx < n) ? d_x[g_tx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncthreads();
    }

    // Warp-level reduce: ALL threads call __shfl_down_sync,
    // only active (tid < offset) threads use the result.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float val = __shfl_down_sync(0xffffffff, sh_mem[tid], offset);
        if (tid < offset)
            sh_mem[tid] += val;
    }

    if (tid == 0)
        atomicAdd(d_y, sh_mem[0]);
}
*/

/* ===========================================================================
 * reduce_cg FIXED
 *
 * Bug: same pattern — tile.shfl_down must be called by every thread
 *      in the tile, but original wrapped it in if(tid < offset).
 *
 * Fix: pull shfl_down outside the if.
 * ===========================================================================
template <typename BLOCK_SZ, typename TILE_SZ>
__global__ void reduce_cg_fixed(float *d_x, int n, float *d_y) {
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid  = threadIdx.x;
    __shared__ float sh_mem[BLOCK_SZ];
    sh_mem[tid] = (g_tx < n) ? d_x[g_tx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= TILE_SZ; offset >>= 1) {
        if (tid < offset)
            sh_mem[tid] += sh_mem[tid + offset];
        __syncthreads();
    }

    cg::thread_block_tile<TILE_SZ> tile =
        cg::tiled_partition<TILE_SZ>(cg::this_thread_block());

    for (int offset = TILE_SZ / 2; offset > 0; offset >>= 1) {
        float val = tile.shfl_down(sh_mem[tid], offset);
        if (tid < offset)
            sh_mem[tid] += val;
    }

    if (tid == 0)
        atomicAdd(d_y, sh_mem[0]);
}
*/
