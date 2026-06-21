/**
 * simpleCUFFT_clean.cu  (NVIDIA original style -- non-causal filter padding)
 *
 * Retains the original NVIDIA cuFFT sample's padding strategy:
 *   - Filter centre is aligned to FFT index 0
 *   - Filter left half wraps to the end of the padded array
 *   - newSize = signalSize + maxRadius (not M+N-1)
 *
 * This is the more complex but historically common DSP approach.
 * See simpleCUFFT_causal.cu for the simpler causal padding alternative.
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
// CPU reference: direct convolution (NVIDIA style, non-causal filter)
// ===========================================================================
static void convolve_cpu(const Complex *signal, int signalSize,
                          const Complex *filterKernel, int filterSize,
                          Complex *result)
{
    int r1 = filterSize / 2;
    int r2 = filterSize - r1;

    for (int i = 0; i < signalSize; ++i) {
        result[i].x = 0.0f;
        result[i].y = 0.0f;
        for (int j = -r2 + 1; j <= r1; ++j) {
            int k = i + j;
            if (k >= 0 && k < signalSize)
                result[i] = ComplexAdd(result[i],
                    ComplexMul(signal[k], filterKernel[r1 - j]));
        }
    }
}

// ===========================================================================
// NVIDIA-style padding (non-causal filter, centre at FFT index 0)
// ===========================================================================
static int pad_data(const Complex *signal, int signalSize,
                     Complex **paddedSignal,
                     const Complex *filterKernel, int filterSize,
                     Complex **paddedFilter)
{
    int r1 = filterSize / 2;
    int r2 = filterSize - r1;
    int newSize = signalSize + r2;

    *paddedSignal = new Complex[newSize];
    for (int i = 0; i < signalSize; ++i)
        (*paddedSignal)[i] = signal[i];
    for (int i = signalSize; i < newSize; ++i) {
        (*paddedSignal)[i].x = 0.0f;
        (*paddedSignal)[i].y = 0.0f;
    }

    // Filter: right half [r1..K-1] at indices [0..r2-1]
    //         zeros in the middle
    //         left half  [0..r1-1]  at indices [N-r1..N-1]
    *paddedFilter = new Complex[newSize];
    for (int i = 0; i < r2; ++i)
        (*paddedFilter)[i] = filterKernel[r1 + i];
    for (int i = r2; i < newSize - r1; ++i) {
        (*paddedFilter)[i].x = 0.0f;
        (*paddedFilter)[i].y = 0.0f;
    }
    for (int i = 0; i < r1; ++i)
        (*paddedFilter)[newSize - r1 + i] = filterKernel[i];

    return newSize;
}

int main()
{
    constexpr int SIGNAL_SIZE        = 50;
    constexpr int FILTER_KERNEL_SIZE = 11;
    constexpr int R1 = FILTER_KERNEL_SIZE / 2;       // minRadius = 5
    constexpr int R2 = FILTER_KERNEL_SIZE - R1;      // maxRadius = 6

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "========================================" << std::endl;
    std::cout << "  simpleCUFFT -- Original NVIDIA style" << std::endl;
    std::cout << "  (non-causal filter, centre @ FFT idx 0)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU    : " << prop.name << std::endl;
    std::cout << "  Signal : " << SIGNAL_SIZE << ", Filter : "
              << FILTER_KERNEL_SIZE << std::endl;
    std::cout << "  minRadius=" << R1 << ", maxRadius=" << R2 << std::endl;
    std::cout << "  newSize= signalSize+maxRadius = " << SIGNAL_SIZE+R2
              << "  (NOT M+N-1=" << SIGNAL_SIZE+FILTER_KERNEL_SIZE-1 << ")"
              << std::endl;
    std::cout << std::endl;

    auto *h_signal = new Complex[SIGNAL_SIZE];
    auto *h_filter = new Complex[FILTER_KERNEL_SIZE];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i].x = dist(rng);
        h_signal[i].y = 0.0f;
    }
    for (int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
        h_filter[i].x = dist(rng);
        h_filter[i].y = 0.0f;
    }

    // CPU reference
    auto *h_conv_ref = new Complex[SIGNAL_SIZE];
    auto t0 = std::chrono::high_resolution_clock::now();
    convolve_cpu(h_signal, SIGNAL_SIZE, h_filter, FILTER_KERNEL_SIZE, h_conv_ref);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "--- CPU reference: " << cpuMs << " ms ---" << std::endl << std::endl;

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
    std::cout << "  Advanced plan work buffer: " << workSize << " bytes" << std::endl << std::endl;

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

    // Verify
    auto *h_conv_gpu = new Complex[newSize];
    CUDA_CHECK(cudaMemcpy(h_conv_gpu, d_signal, memSize, cudaMemcpyDeviceToHost));

    double errorNorm = 0.0, refNorm = 0.0;
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
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
    std::cout << "  Rel error: " << std::scientific << errorNorm/refNorm << std::fixed << std::endl;
    std::cout << "  Result   : " << (pass ? "PASS" : "FAIL") << std::endl;

    // Sample output
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
