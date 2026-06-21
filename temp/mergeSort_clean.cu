/**
 * mergeSort_clean.cu
 *
 * Single-file version of NVIDIA's mergeSort sample.
 *
 * Based on: "Designing efficient sorting algorithms for manycore GPUs"
 * by Nadathur Satish, Mark Harris, and Michael Garland
 * http://mgarland.org/files/papers/gpusort-ipdps09.pdf
 *
 * Algorithm overview:
 *   This is a bottom-up merge sort operating on key-value pairs. Each
 *   element has a key (sort criteria) and a value (payload, preserved as
 *   stable sort). The algorithm proceeds in three phases per doubling:
 *
 *   Phase 1 - Bottom-level sort:
 *     Sort individual chunks of SHARED_SIZE_LIMIT (1024) elements using
 *     a binary-search-based merge in shared memory.  This replaces the
 *     initial O(n²) bubble sort from the CPU version.
 *
 *   Phase 2 - Larger merges (stride = 1024, 2048, 4096, ... up to N):
 *     Step A: Generate sample ranks -- For each pair of sorted segments
 *             (each of length 'stride'), compute the rank (sorted position)
 *             of SAMPLE_STRIDE-spaced samples using binary search.
 *     Step B: Merge ranks -- Merge the two rank arrays to find elementary
 *             interval boundaries. Each interval has ≤ SAMPLE_STRIDE elements.
 *     Step C: Merge intervals -- For each interval, load source data into
 *             shared memory, perform binary-search merge, store back.
 *
 * Key design choices:
 *   - Shared memory limit: 1024 elements (SHARED_SIZE_LIMIT)
 *   - Sample stride: 128 elements (balances parallelism vs overhead)
 *   - Double-buffering: ping-pong between d_Dst/d_Buf across stages
 *   - sortDir = 1 means ascending, 0 means descending
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Error-check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[ERROR] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
typedef unsigned int uint;
static const uint SHARED_SIZE_LIMIT = 1024U;
static const uint SAMPLE_STRIDE     = 128;
static const uint MAX_SAMPLE_COUNT  = 32768;

// ===========================================================================
// Utility
// ===========================================================================
static __host__ __device__ uint iDivUp(uint a, uint b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static __host__ __device__ uint getSampleCount(uint dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

static __host__ __device__ uint umin(uint a, uint b)
{
    return (a <= b) ? a : b;
}

#define W (sizeof(uint) * 8)

// Next power of two using __clz (count leading zeros) on device,
// fallback on host.
static __host__ __device__ uint nextPowerOfTwo(uint x)
{
#ifdef __CUDA_ARCH__
    return 1U << (W - __clz(x - 1));
#else
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
#endif
}

// ===========================================================================
// Binary search helpers (device + optionally host)
// ===========================================================================
template <uint sortDir>
static inline __device__ uint binarySearchInclusive(
    uint val, uint *data, uint L, uint stride)
{
    if (L == 0) return 0;
    uint pos = 0;
    for (; stride > 0; stride >>= 1) {
        uint newPos = umin(pos + stride, L);
        if ((sortDir && (data[newPos - 1] <= val))
         || (!sortDir && (data[newPos - 1] >= val)))
            pos = newPos;
    }
    return pos;
}

template <uint sortDir>
static inline __device__ uint binarySearchExclusive(
    uint val, uint *data, uint L, uint stride)
{
    if (L == 0) return 0;
    uint pos = 0;
    for (; stride > 0; stride >>= 1) {
        uint newPos = umin(pos + stride, L);
        if ((sortDir && (data[newPos - 1] < val))
         || (!sortDir && (data[newPos - 1] > val)))
            pos = newPos;
    }
    return pos;
}

// ===========================================================================
// Phase 1: Bottom-level merge sort (shared memory)
// ===========================================================================
//   Each block handles SHARED_SIZE_LIMIT = 1024 consecutive elements.
//   Thread count = SHARED_SIZE_LIMIT/2 = 512 (each thread loads 2 elements).
//
//   Algorithm: iterative binary-search merge within shared memory.
//   For stride = 1, 2, 4, ... up to arrayLength/2:
//     - Each thread reads two elements from shared memory, computes
//       their target positions (using binary search), and swaps.
//   After all iterations, the chunk is sorted in shared memory,
//   then written back to global memory.
// ===========================================================================
template <uint sortDir>
__global__ void mergeSortSharedKernel(
    uint *d_DstKey, uint *d_DstVal,
    uint *d_SrcKey, uint *d_SrcVal,
    uint arrayLength)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    // Each thread loads 2 elements into shared memory
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT;
    uint tid = threadIdx.x;
    s_key[tid]                                  = d_SrcKey[tid];
    s_val[tid]                                  = d_SrcVal[tid];
    s_key[tid + (SHARED_SIZE_LIMIT / 2)]        = d_SrcKey[tid + (SHARED_SIZE_LIMIT / 2)];
    s_val[tid + (SHARED_SIZE_LIMIT / 2)]        = d_SrcVal[tid + (SHARED_SIZE_LIMIT / 2)];

    for (uint stride = 1; stride < arrayLength; stride <<= 1) {
        uint lPos    = tid & (stride - 1);
        uint *baseKey = s_key + 2 * (tid - lPos);
        uint *baseVal = s_val + 2 * (tid - lPos);

        cg::sync(cta);
        uint keyA = baseKey[lPos + 0];
        uint valA = baseVal[lPos + 0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive<sortDir>(
                        keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<sortDir>(
                        keyB, baseKey + 0, stride, stride) + lPos;

        cg::sync(cta);
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    // Write sorted chunk back to global memory
    cg::sync(cta);
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT;
    d_DstKey[tid]                                  = s_key[tid];
    d_DstVal[tid]                                  = s_val[tid];
    d_DstKey[tid + (SHARED_SIZE_LIMIT / 2)]        = s_key[tid + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[tid + (SHARED_SIZE_LIMIT / 2)]        = s_val[tid + (SHARED_SIZE_LIMIT / 2)];
}

static void mergeSortShared(
    uint *d_DstKey, uint *d_DstVal,
    uint *d_SrcKey, uint *d_SrcVal,
    uint batchSize, uint arrayLength, uint sortDir)
{
    if (arrayLength < 2) return;
    uint blockCount  = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    if (sortDir)
        mergeSortSharedKernel<1U><<<blockCount, threadCount>>>(
            d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
    else
        mergeSortSharedKernel<0U><<<blockCount, threadCount>>>(
            d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Phase 2A: Generate sample ranks
// ===========================================================================
//   For each pair of segments (A of length stride, B of length min(stride, N-)), //   sample every SAMPLE_STRIDE elements from A and B. For each sample,//
//   compute its rank (position) in the OTHER segment using binary search.
//   Result: ranksA and ranksB arrays hold the sorted positions of samples.
// ===========================================================================
template <uint sortDir>
__global__ void generateSampleRanksKernel(
    uint *d_RanksA, uint *d_RanksB,
    uint *d_SrcKey, uint stride, uint N, uint threadCount)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= threadCount) return;

    const uint i           = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint segmentSamplesA  = getSampleCount(segmentElementsA);
    const uint segmentSamplesB  = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA) {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<sortDir>(
            d_SrcKey[i * SAMPLE_STRIDE],
            d_SrcKey + stride,
            segmentElementsB,
            nextPowerOfTwo(segmentElementsB));
    }
    if (i < segmentSamplesB) {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<sortDir>(
            d_SrcKey[stride + i * SAMPLE_STRIDE],
            d_SrcKey + 0,
            segmentElementsA,
            nextPowerOfTwo(segmentElementsA));
    }
}

static void generateSampleRanks(
    uint *d_RanksA, uint *d_RanksB,
    uint *d_SrcKey, uint stride, uint N, uint sortDir)
{
    uint lastSegmentElements = N % (2 * stride);
    uint threadCount = (lastSegmentElements > stride)
        ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
        : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);
    if (sortDir)
        generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256>>>(
            d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
    else
        generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256>>>(
            d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Phase 2B: Merge ranks and indices → elementary interval boundaries
// ===========================================================================
__global__ void mergeRanksAndIndicesKernel(
    uint *d_Limits, uint *d_Ranks,
    uint stride, uint N, uint threadCount)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= threadCount) return;

    const uint i           = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint segmentSamplesA  = getSampleCount(segmentElementsA);
    const uint segmentSamplesB  = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA) {
        uint dstPos = binarySearchExclusive<1U>(
            d_Ranks[i], d_Ranks + segmentSamplesA,
            segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }
    if (i < segmentSamplesB) {
        uint dstPos = binarySearchInclusive<1U>(
            d_Ranks[segmentSamplesA + i], d_Ranks,
            segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
    }
}

static void mergeRanksAndIndices(
    uint *d_LimitsA, uint *d_LimitsB,
    uint *d_RanksA, uint *d_RanksB,
    uint stride, uint N)
{
    uint lastSegmentElements = N % (2 * stride);
    uint threadCount = (lastSegmentElements > stride)
        ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
        : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsA, d_RanksA, stride, N, threadCount);
    CUDA_CHECK(cudaGetLastError());

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsB, d_RanksB, stride, N, threadCount);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Phase 2C: Merge elementary intervals
// ===========================================================================
//   Each interval has ≤ SAMPLE_STRIDE elements.
//   Threads = SAMPLE_STRIDE (128), each thread loads 1 element from A or B
//   (whichever side is assigned) into shared memory. Then a binary-search
//   merge determines final positions and writes back to global memory.
// ===========================================================================
template <uint sortDir>
inline __device__ void mergeFunc(
    uint *dstKey, uint *dstVal,
    uint *srcAKey, uint *srcAVal,
    uint *srcBKey, uint *srcBVal,
    uint lenA, uint nPowTwoLenA,
    uint lenB, uint nPowTwoLenB,
    cg::thread_block cta)
{
    uint tid = threadIdx.x;

    uint keyA, valA, keyB, valB, dstPosA, dstPosB;

    if (tid < lenA) {
        keyA    = srcAKey[tid];
        valA    = srcAVal[tid];
        dstPosA = binarySearchExclusive<sortDir>(
                      keyA, srcBKey, lenB, nPowTwoLenB) + tid;
    }
    if (tid < lenB) {
        keyB    = srcBKey[tid];
        valB    = srcBVal[tid];
        dstPosB = binarySearchInclusive<sortDir>(
                      keyB, srcAKey, lenA, nPowTwoLenA) + tid;
    }

    cg::sync(cta);

    if (tid < lenA) {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }
    if (tid < lenB) {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}

template <uint sortDir>
__global__ void mergeElementaryIntervalsKernel(
    uint *d_DstKey, uint *d_DstVal,
    uint *d_SrcKey, uint *d_SrcVal,
    uint *d_LimitsA, uint *d_LimitsB,
    uint stride, uint N)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_key[2 * SAMPLE_STRIDE];
    __shared__ uint s_val[2 * SAMPLE_STRIDE];

    uint tid = threadIdx.x;

    const uint intervalI   = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (tid == 0) {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint segmentSamplesA  = getSampleCount(segmentElementsA);
        uint segmentSamplesB  = getSampleCount(segmentElementsB);
        uint segmentSamples   = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        uint endSrcA = (intervalI + 1 < segmentSamples)
                         ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples)
                         ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    // Load source data into shared memory
    cg::sync(cta);
    if (tid < lenSrcA) {
        s_key[tid] = d_SrcKey[0 + startSrcA + tid];
        s_val[tid] = d_SrcVal[0 + startSrcA + tid];
    }
    if (tid < lenSrcB) {
        s_key[tid + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + tid];
        s_val[tid + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + tid];
    }

    // Binary-search merge in shared memory
    cg::sync(cta);
    mergeFunc<sortDir>(
        s_key, s_val,
        s_key + 0, s_val + 0,
        s_key + SAMPLE_STRIDE, s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE,
        cta);

    // Store merged result back
    cg::sync(cta);
    if (tid < lenSrcA) {
        d_DstKey[startDstA + tid] = s_key[tid];
        d_DstVal[startDstA + tid] = s_val[tid];
    }
    if (tid < lenSrcB) {
        d_DstKey[startDstB + tid] = s_key[lenSrcA + tid];
        d_DstVal[startDstB + tid] = s_val[lenSrcA + tid];
    }
}

static void mergeElementaryIntervals(
    uint *d_DstKey, uint *d_DstVal,
    uint *d_SrcKey, uint *d_SrcVal,
    uint *d_LimitsA, uint *d_LimitsB,
    uint stride, uint N, uint sortDir)
{
    uint lastSegmentElements = N % (2 * stride);
    uint mergePairs = (lastSegmentElements > stride)
        ? getSampleCount(N)
        : (N - lastSegmentElements) / SAMPLE_STRIDE;

    if (sortDir)
        mergeElementaryIntervalsKernel<1U>
            <<<mergePairs, SAMPLE_STRIDE>>>(
                d_DstKey, d_DstVal, d_SrcKey, d_SrcVal,
                d_LimitsA, d_LimitsB, stride, N);
    else
        mergeElementaryIntervalsKernel<0U>
            <<<mergePairs, SAMPLE_STRIDE>>>(
                d_DstKey, d_DstVal, d_SrcKey, d_SrcVal,
                d_LimitsA, d_LimitsB, stride, N);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Top-level merge sort driver
// ===========================================================================
//   Double-buffer ping-pong: d_Dst[Key|Val] and d_Buf[Key|Val] alternate
//   across stages to avoid extra copies.
// ===========================================================================
static void mergeSort(
    uint *d_DstKey, uint *d_DstVal,
    uint *d_BufKey, uint *d_BufVal,
    uint *d_SrcKey, uint *d_SrcVal,
    uint N, uint sortDir,
    uint *d_RanksA, uint *d_RanksB,
    uint *d_LimitsA, uint *d_LimitsB)
{
    // Count how many doubling stages we need
    uint stageCount = 0;
    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1)
        stageCount++;

    // Determine which buffer is input and which is output
    uint *ikey, *ival, *okey, *oval;
    if (stageCount & 1) {
        ikey = d_BufKey; ival = d_BufVal;
        okey = d_DstKey; oval = d_DstVal;
    } else {
        ikey = d_DstKey; ival = d_DstVal;
        okey = d_BufKey; oval = d_BufVal;
    }

    // Phase 1: bottom-level shared-memory merge sort
    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal,
                    N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir);

    // Phase 2: double-buffered merge stages
    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1) {
        uint lastSegmentElements = N % (2 * stride);

        // A) Generate sample ranks
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

        // B) Merge ranks to get interval boundaries
        mergeRanksAndIndices(d_LimitsA, d_LimitsB,
                              d_RanksA, d_RanksB, stride, N);

        // C) Merge each elementary interval
        mergeElementaryIntervals(okey, oval, ikey, ival,
                                  d_LimitsA, d_LimitsB,
                                  stride, N, sortDir);

        // If the last segment is a lone array (not a pair), pass it through
        if (lastSegmentElements <= stride) {
            CUDA_CHECK(cudaMemcpy(
                okey + (N - lastSegmentElements),
                ikey + (N - lastSegmentElements),
                lastSegmentElements * sizeof(uint),
                cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(
                oval + (N - lastSegmentElements),
                ival + (N - lastSegmentElements),
                lastSegmentElements * sizeof(uint),
                cudaMemcpyDeviceToDevice));
        }

        // Swap buffers
        uint *t;
        t = ikey; ikey = okey; okey = t;
        t = ival; ival = oval; oval = t;
    }
}

// ===========================================================================
// Validation
// ===========================================================================
static bool validateSortedKeys(
    const uint *resKey, const uint *srcKey,
    uint batchSize, uint arrayLength, uint numValues, uint sortDir)
{
    if (arrayLength < 2) return true;

    printf("...inspecting keys array: ");

    uint *srcHist = new uint[numValues];
    uint *resHist = new uint[numValues];

    bool ok = true;

    for (uint j = 0; j < batchSize; j++) {
        // Zero histograms for this batch
        for (uint i = 0; i < numValues; i++) srcHist[i] = 0;
        for (uint i = 0; i < numValues; i++) resHist[i] = 0;

        // Build histograms
        for (uint i = 0; i < arrayLength; i++) {
            if (srcKey[i] < numValues && resKey[i] < numValues) {
                srcHist[srcKey[i]]++;
                resHist[resKey[i]]++;
            } else {
                fprintf(stderr, "***Set %u key values out of range***\n", j);
                ok = false;
                goto done;
            }
        }

        // Compare histograms: same elements must exist in source and result
        for (uint i = 0; i < numValues; i++) {
            if (srcHist[i] != resHist[i]) {
                fprintf(stderr,
                    "***Set %u source/result key histograms mismatch***\n", j);
                ok = false;
                goto done;
            }
        }

        // Check sort order
        for (uint i = 0; i < arrayLength - 1; i++) {
            if ((sortDir && (resKey[i] > resKey[i + 1]))
             || (!sortDir && (resKey[i] < resKey[i + 1]))) {
                fprintf(stderr,
                    "***Set %u result key array not ordered***\n", j);
                ok = false;
                goto done;
            }
        }

        srcKey += arrayLength;
        resKey += arrayLength;
    }

done:
    delete[] resHist;
    delete[] srcHist;
    printf(ok ? "OK\n" : "FAIL\n");
    return ok;
}

static bool validateSortedValues(
    const uint *resKey, const uint *resVal,
    const uint *srcKey, uint batchSize, uint arrayLength)
{
    printf("...inspecting keys and values array: ");

    bool correct = true;
    bool stable  = true;

    for (uint i = 0; i < batchSize; i++) {
        for (uint j = 0; j < arrayLength; j++) {
            // Value points back to source key → key should match
            if (resKey[j] != srcKey[resVal[j]])
                correct = false;
            // Same key, earlier index → value should be smaller (stable sort)
            if ((j < arrayLength - 1) && (resKey[j] == resKey[j + 1])
                && (resVal[j] > resVal[j + 1]))
                stable = false;
        }
        resKey += arrayLength;
        resVal += arrayLength;
    }

    printf(correct ? "OK\n" : "***corrupted!***\n");
    printf(stable
            ? "...stability property: stable!\n"
            : "...stability property: NOT stable\n");
    return correct;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    // ------------------------------------------------------------------
    // 1. Pick device 0
    // ------------------------------------------------------------------
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        printf("No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: \"%s\", SM %d.%d\n", prop.name, prop.major, prop.minor);

    // ------------------------------------------------------------------
    // 2. Parameters
    // ------------------------------------------------------------------
    const uint N       = 4 * 1024 * 1024;  // 4M elements
    const uint DIR     = 1;                // 1 = ascending
    const uint numVals = 65536;            // key range [0, numVals)

    printf("[mergeSort] Sorting %u elements, key range [0, %u)...\n\n",
           N, numVals);

    // ------------------------------------------------------------------
    // 3. Allocate + initialize host memory
    // ------------------------------------------------------------------
    uint *h_SrcKey = new uint[N];
    uint *h_SrcVal = new uint[N];
    uint *h_DstKey = new uint[N];
    uint *h_DstVal = new uint[N];

    // Random keys, sequential values (for stability check)
    std::mt19937 rng(2009);
    std::uniform_int_distribution<uint> dist(0, numVals - 1);

    for (uint i = 0; i < N; i++) {
        h_SrcKey[i] = dist(rng);
        h_SrcVal[i] = i;     // sequential values → stable sort test
    }

    printf("Allocating and initializing host arrays: done\n");

    // ------------------------------------------------------------------
    // 4. Allocate + init device memory
    // ------------------------------------------------------------------
    printf("Allocating device arrays...\n");

    uint *d_DstKey = nullptr, *d_DstVal = nullptr;
    uint *d_BufKey = nullptr, *d_BufVal = nullptr;
    uint *d_SrcKey = nullptr, *d_SrcVal = nullptr;

    CUDA_CHECK(cudaMalloc(&d_DstKey, N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_DstVal, N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_BufKey, N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_BufVal, N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_SrcKey, N * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_SrcVal, N * sizeof(uint)));

    CUDA_CHECK(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint),
                          cudaMemcpyHostToDevice));

    // Allocate rank/limit buffers for merge stages
    uint *d_RanksA = nullptr, *d_RanksB = nullptr;
    uint *d_LimitsA = nullptr, *d_LimitsB = nullptr;

    CUDA_CHECK(cudaMalloc(&d_RanksA, MAX_SAMPLE_COUNT * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_RanksB, MAX_SAMPLE_COUNT * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(uint)));

    // ------------------------------------------------------------------
    // 5. Warm-up + timed sort
    // ------------------------------------------------------------------
    printf("Initializing GPU merge sort...\n");

    // Warm-up
    mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal,
              d_SrcKey, d_SrcVal, N, DIR,
              d_RanksA, d_RanksB, d_LimitsA, d_LimitsB);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-load source data for timed run
    CUDA_CHECK(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint),
                          cudaMemcpyHostToDevice));

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running GPU merge sort...\n");
    CUDA_CHECK(cudaEventRecord(start));
    mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal,
              d_SrcKey, d_SrcVal, N, DIR,
              d_RanksA, d_RanksB, d_LimitsA, d_LimitsB);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("  Time: %.3f ms\n\n", ms);

    // ------------------------------------------------------------------
    // 6. Copy result back + verify
    // ------------------------------------------------------------------
    printf("Reading back GPU merge sort results...\n");
    CUDA_CHECK(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint),
                          cudaMemcpyDeviceToHost));

    printf("Inspecting the results...\n");
    bool keysOk = validateSortedKeys(
        h_DstKey, h_SrcKey, 1, N, numVals, DIR);
    bool valsOk = validateSortedValues(
        h_DstKey, h_DstVal, h_SrcKey, 1, N);

    // ------------------------------------------------------------------
    // 7. Cleanup
    // ------------------------------------------------------------------
    printf("\nShutting down...\n");
    CUDA_CHECK(cudaFree(d_SrcVal));
    CUDA_CHECK(cudaFree(d_SrcKey));
    CUDA_CHECK(cudaFree(d_BufVal));
    CUDA_CHECK(cudaFree(d_BufKey));
    CUDA_CHECK(cudaFree(d_DstVal));
    CUDA_CHECK(cudaFree(d_DstKey));
    CUDA_CHECK(cudaFree(d_RanksA));
    CUDA_CHECK(cudaFree(d_RanksB));
    CUDA_CHECK(cudaFree(d_LimitsA));
    CUDA_CHECK(cudaFree(d_LimitsB));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_DstVal;
    delete[] h_DstKey;
    delete[] h_SrcVal;
    delete[] h_SrcKey;

    bool passed = keysOk && valsOk;
    printf(passed ? "Test PASSED\n" : "Test FAILED\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
