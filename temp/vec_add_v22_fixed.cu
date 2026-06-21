//
// Fixed vec_add_v22 — grid-stride loop + float4
// BUG in original: loop body used `idx` instead of `i`,
// so only the first stride-iteration actually wrote data.
//
__global__ void vec_add_v22(const float * __restrict__ a,
                             const float * __restrict__ b,
                             int n,
                             float * __restrict__ c)
{
    int stride = gridDim.x * blockDim.x * 4;
    int idx    = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    for (int i = idx; i < n; i += stride) {
        int remaining = n - i;
        if (remaining >= 4) {
            float4 tmp_a = LD_FLOAT4(a[i]);
            float4 tmp_b = LD_FLOAT4(b[i]);
            float4 tmp_c;
            tmp_c.x = tmp_a.x + tmp_b.x;
            tmp_c.y = tmp_a.y + tmp_b.y;
            tmp_c.z = tmp_a.z + tmp_b.z;
            tmp_c.w = tmp_a.w + tmp_b.w;
            ST_FLOAT4(c[i]) = tmp_c;
        } else {
            #pragma unroll
            for (int k = 0; k < remaining; k++)
                c[i + k] = a[i + k] + b[i + k];
        }
    }
}
