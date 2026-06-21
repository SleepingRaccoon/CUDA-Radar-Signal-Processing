#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// CPU reference
__host__ void reduce_cpu(const float *d_x, const int n, float *d_y);

// Global-memory reduce (modifies input)
__global__ void reduce_global(float *d_x, const int n, float *d_y);

// Shared-memory reduce (block-based)
// Uses dynamic shared memory -- smem size passed at launch <<<grid, block, smem>>>
__global__ void reduce_shared(const float *d_x, const int n, float *d_y);

// Shared-memory reduce (grid-stride loop, fewer blocks)
__global__ void reduce_shared_stride(const float *d_x, const int n, float *d_y);

// Shared-memory reduce with __syncwarp for the last warp
__global__ void reduce_syncwrap(const float *d_x, const int n, float *d_y);

// Shared-memory reduce with __shfl_down_sync for the last log2(32) rounds
__global__ void reduce_shfl(const float *d_x, const int n, float *d_y);

// Cooperative Groups reduce
// TILE_SZ is a template parameter (tiled_partition needs it at compile time)
template <int TILE_SZ>
__global__ void reduce_cg(const float *d_x, const int n, float *d_y)
{
    int g_tx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sh_mem[];
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
