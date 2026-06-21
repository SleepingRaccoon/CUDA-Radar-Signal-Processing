/**
 * simpleCUFFT_causal.cu
 *
 * FFT-based 1D convolution using cuFFT -- CAUSAL (simple) padding.
 *
 * Both signal and filter are treated as CAUSAL sequences starting at
 * index 0. Zero-pad to M+N-1, FFT both, multiply, IFFT. The full
 * linear convolution result is at indices [0 .. M+N-2].
 *
 * Contrast with simpleCUFFT_clean.cu which uses the NVIDIA-style
 * non-causal filter padding (centre-aligned to FFT index 0).
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__         \
                      << "  " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CUFFT_CHECK(call)                                                   \
    do {                                                                    \
        cufftResult res = (call);                                           \
        if (res != CUFFT_SUCCESS) {                                         \
            std::cerr << "[CUFFT ERROR] " << __FILE__ << ":" << __LINE__   \
                      << "  status=" << static_cast<int>(res) << std::endl; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

using Complex = float2;

static __host__ __device__ inline Complex ComplexAdd(Complex a, Complex b)
{
    return {a.x + b.x, a.y + b.y};
}

static __host__ __device__ inline Complex ComplexMul(Complex a, Complex b)
{
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

static __host__ __device__ inline Complex ComplexScale(Complex a, float s)
{
    return {s * a.x, s * a.y};
}

__global__ void ComplexPointwiseMulAndScale(
    Complex *a, const Complex *b, int size, float scale)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < size; i += stride)
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}

// ===========================================================================
// CPU reference: standard linear convolution
//   output[k] = sum_i signal[i] * filter[k-i]
//   output length = signalSize + filterSize - 1
// ===========================================================================
static void convolve_cpu(const Complex *signal, int signalSize,
                          const Complex *filterKernel, int filterSize,
                          Complex *result)
{
    int outLen = signalSize + filterSize - 1;
    for (int k = 0; k < outLen; ++k) {
        result[k].x = result[k].y = 0.0f;
        for (int i = 0; i < signalSize; ++i) {
            int j = k - i;
            if (j >= 0 && j < filterSize)
                result[k] = ComplexAdd(result[k],
                    ComplexMul(signal[i], filterKernel[j]));
        }
    }
}

// ===========================================================================
// Causal padding: both signal and filter at index 0, zero-fill to end
// ===========================================================================
static int pad_data(const Complex *signal, int signalSize,
                     Complex **paddedSignal,
                     const Complex *filterKernel, int filterSize,
                     Complex **paddedFilter)
{
    int newSize = signalSize + filterSize - 1;  // M + N - 1

    // Signal: copy, then zero-fill
    *paddedSignal = new Complex[newSize];
    for (int i = 0; i < signalSize; ++i)
        (*paddedSignal)[i] = signal[i];
    for (int i = signalSize; i < newSize; ++i)
        (*paddedSignal)[i] = {0.0f, 0.0f};

    // Filter: copy, then zero-fill  (no rearrangement needed)
    *paddedFilter = new Complex[newSize];
    for (int i = 0; i < filterSize; ++i)
        (*paddedFilter)[i] = filterKernel[i];
    for (int i = filterSize; i < newSize; ++i)
        (*paddedFilter)[i] = {0.0f, 0.0f};

    return newSize;
}

int main()
{
    constexpr int SIGNAL_SIZE        = 50;
    constexpr int FILTER_KERNEL_SIZE = 11;
    constexpr int PADDED_SIZE = SIGNAL_SIZE + FILTER_KERNEL_SIZE - 1; // M+N-1

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "========================================" << std::endl;
    std::cout << "  simpleCUFFT -- Causal (simple) style" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU    : " << prop.name << std::endl;
    std::cout << "  Signal : " << SIGNAL_SIZE << ", Filter : "
              << FILTER_KERNEL_SIZE << std::endl;
    std::cout << "  Padded : " << PADDED_SIZE << " (= "
              << SIGNAL_SIZE << " + " << FILTER_KERNEL_SIZE << " - 1)"
              << std::endl;
    std::cout << std::endl;

    auto *h_signal = new Complex[SIGNAL_SIZE];
    auto *h_filter = new Complex[FILTER_KERNEL_SIZE];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < SIGNAL_SIZE; ++i)
        h_signal[i] = {dist(rng), 0.0f};
    for (int i = 0; i < FILTER_KERNEL_SIZE; ++i)
        h_filter[i] = {dist(rng), 0.0f};

    // CPU reference
    int outLen = SIGNAL_SIZE + FILTER_KERNEL_SIZE - 1;
    auto *h_conv_ref = new Complex[outLen];
    auto t0 = std::chrono::high_resolution_clock::now();
    convolve_cpu(h_signal, SIGNAL_SIZE, h_filter, FILTER_KERNEL_SIZE, h_conv_ref);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  CPU ref: " << cpuMs << " ms" << std::endl << std::endl;

    // Pad
    Complex *h_padded_signal = nullptr, *h_padded_filter = nullptr;
    int newSize = pad_data(h_signal, SIGNAL_SIZE, &h_padded_signal,
                           h_filter, FILTER_KERNEL_SIZE, &h_padded_filter);
    int memSize = newSize * (int)sizeof(Complex);

    // Device memory
    Complex *d_signal = nullptr, *d_filter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, memSize));
    CUDA_CHECK(cudaMalloc(&d_filter, memSize));
    CUDA_CHECK(cudaMemcpy(d_signal, h_padded_signal, memSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_padded_filter, memSize, cudaMemcpyHostToDevice));

    // cuFFT plans
    cufftHandle plan, planAdv;
    size_t workSize = 0;
    long long nLong = newSize;
    CUFFT_CHECK(cufftPlan1d(&plan, newSize, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftCreate(&planAdv));
    CUFFT_CHECK(cufftXtMakePlanMany(planAdv, 1, &nLong,
        nullptr, 1, 1, CUDA_C_32F, nullptr, 1, 1, CUDA_C_32F, 1, &workSize, CUDA_C_32F));
    std::cout << "  Advanced plan work buffer: " << workSize << " bytes" << std::endl;
    std::cout << std::endl;

    // Warm-up
    CUFFT_CHECK(cufftExecC2C(planAdv, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(planAdv, (cufftComplex*)d_filter, (cufftComplex*)d_filter, CUFFT_FORWARD));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_signal, h_padded_signal, memSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_padded_filter, memSize, cudaMemcpyHostToDevice));

    // Timed forward FFTs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float fftMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fftMs, start, stop));
    std::cout << "  Signal FFT : " << fftMs << " ms" << std::endl;

    CUDA_CHECK(cudaEventRecord(start));
    CUFFT_CHECK(cufftExecC2C(planAdv, (cufftComplex*)d_filter, (cufftComplex*)d_filter, CUFFT_FORWARD));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&fftMs, start, stop));
    std::cout << "  Filter FFT : " << fftMs << " ms" << std::endl << std::endl;

    // Multiply + inverse FFT
    ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter, newSize, 1.0f/newSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_INVERSE));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&fftMs, start, stop));
    std::cout << "  Inverse FFT: " << fftMs << " ms" << std::endl << std::endl;

    // Verify (full M+N-1 samples)
    auto *h_conv_gpu = new Complex[newSize];
    CUDA_CHECK(cudaMemcpy(h_conv_gpu, d_signal, memSize, cudaMemcpyDeviceToHost));

    double errorNorm = 0.0, refNorm = 0.0;
    for (int i = 0; i < outLen; ++i) {
        double dx = (double)h_conv_ref[i].x - (double)h_conv_gpu[i].x;
        double dy = (double)h_conv_ref[i].y - (double)h_conv_gpu[i].y;
        errorNorm += dx*dx + dy*dy;
        refNorm   += (double)h_conv_ref[i].x * h_conv_ref[i].x
                   + (double)h_conv_ref[i].y * h_conv_ref[i].y;
    }
    errorNorm = std::sqrt(errorNorm);
    refNorm   = std::sqrt(refNorm);
    bool pass = (errorNorm / refNorm < 1.0e-4);

    std::cout << "--- Verification ---" << std::endl;
    std::cout << "  First index: 0 + 0 = 0  (signal[0] * filter[0])" << std::endl;
    std::cout << "  Last  index: " << outLen - 1 << std::endl;
    std::cout << "  Rel error: " << std::scientific << errorNorm/refNorm << std::fixed << std::endl;
    std::cout << "  Result   : " << (pass ? "PASS" : "FAIL") << std::endl;

    std::cout << std::endl << "  First 8 samples:" << std::endl;
    for (int i = 0; i < 8; ++i)
        std::cout << "    [" << i << "] CPU=" << h_conv_ref[i].x
                  << "  GPU=" << h_conv_gpu[i].x
                  << "  imag=" << h_conv_gpu[i].y << std::endl;

    // Cleanup
    delete[] h_signal; delete[] h_filter; delete[] h_conv_ref;
    delete[] h_padded_signal; delete[] h_padded_filter; delete[] h_conv_gpu;
    CUDA_CHECK(cudaFree(d_signal)); CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUFFT_CHECK(cufftDestroy(plan)); CUFFT_CHECK(cufftDestroy(planAdv));

    std::cout << std::endl << (pass ? "Test PASSED" : "Test FAILED") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
