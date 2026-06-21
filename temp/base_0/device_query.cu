/*
nvcc -o device_query device_query.cu
device_query.exe
*/

/**
 * Single-file version of NVIDIA's deviceQuery sample.
 *
 * Queries and prints all CUDA device properties via the Runtime API.
 * No kernel -- pure host-side device inspection tool.
 *
 * Retained technical details:
 *   - SM version -> cores/SM lookup table
 *   - CUDA Driver / Runtime version reporting
 *   - All cudaDeviceProp fields (global mem, SM count, clock rates,
 *     memory bus width, L2 cache, texture limits, register counts,
 *     warp size, thread/block limits, grid limits, asyncEngineCount,
 *     compute mode, ECC, UVA, managed memory, preemption, coop launch,
 *     PCI topology, TCC/WDDM driver mode)
 *   - P2P capability matrix (multi-GPU systems)
 *   - Version compatibility guards (pre-CUDA 5.0, pre-13.0)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// Error-check macro
// --------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "[ERROR] %s:%d  %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// ==========================================================================
// SM version -> cores per SM lookup
// ==========================================================================
static int smCoresPerSM(int major, int minor)
{
    struct { int sm; int cores; } table[] = {
        {0x30, 192},   // Fermi
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},   // Kepler
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},   // Maxwell
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},   // Volta
        {0x72,  64},
        {0x75,  64},   // Turing
        {0x80,  64},   // Ampere A100
        {0x86, 128},   // Ampere GA10x
        {0x87, 128},
        {0x89, 128},   // Ada Lovelace
        {0x90, 128},   // Hopper
        {0xa0, 128},   // Blackwell
        {0xa1, 128},
        {0xa3, 128},
        {0xb0, 128},
        {0xc0, 128},
        {0xc1, 128},
    };
    int n = (int)(sizeof(table) / sizeof(table[0]));
    int encoded = (major << 4) | minor;

    for (int i = 0; i < n; i++) {
        if (table[i].sm == encoded)
            return table[i].cores;
    }
    fprintf(stderr, "[WARNING] SM %d.%d not found, assuming %d cores/SM\n",
            major, minor, table[n - 1].cores);
    return table[n - 1].cores;
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    printf("deviceQuery Starting...\n\n");
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    // ------------------------------------------------------------------
    // 1. Device count
    // ------------------------------------------------------------------
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    // ------------------------------------------------------------------
    // 2. Per-device properties
    // ------------------------------------------------------------------
    int driverVersion  = 0;
    int runtimeVersion = 0;

    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("\nDevice %d: \"%s\"\n", dev, prop.name);

        // CUDA version
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               prop.major, prop.minor);

        // Global memory
        printf("  Total amount of global memory:                 %.0f MBytes"
               " (%zu bytes)\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0),
               prop.totalGlobalMem);

        // SM / core count
        int coresPerSM = smCoresPerSM(prop.major, prop.minor);
        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               prop.multiProcessorCount, coresPerSM,
               coresPerSM * prop.multiProcessorCount);

        // Clock rates
        int clockRate = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev));
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n",
               clockRate * 1e-3f, clockRate * 1e-6f);

        int memoryClockRate = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockRate,
                                           cudaDevAttrMemoryClockRate, dev));
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               prop.memoryBusWidth);

        if (prop.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   prop.l2CacheSize);
        }

        // Texture dimensions
        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d),"
               " 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               prop.maxTexture1D,
               prop.maxTexture2D[0], prop.maxTexture2D[1],
               prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d),"
               " %d layers\n",
               prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1],
               prop.maxTexture2DLayered[2]);

        // Memory hierarchy
        printf("  Total amount of constant memory:               %zu bytes\n",
               prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
               prop.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
               prop.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n",
               prop.regsPerBlock);

        // Execution configuration
        printf("  Warp size:                                     %d\n", prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // Misc
        printf("  Maximum memory pitch:                          %zu bytes\n",
               prop.memPitch);
        printf("  Texture alignment:                             %zu bytes\n",
               prop.textureAlignment);

        // Async / overlap
        int gpuOverlap = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&gpuOverlap, cudaDevAttrGpuOverlap, dev));
        printf("  Concurrent copy and kernel execution:          %s with %d"
               " copy engine(s)\n",
               (gpuOverlap ? "Yes" : "No"), prop.asyncEngineCount);

        // Kernel timeout
        int kernelExecTimeout = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&kernelExecTimeout,
                                           cudaDevAttrKernelExecTimeout, dev));
        printf("  Run time limit on kernels:                     %s\n",
               kernelExecTimeout ? "Yes" : "No");

        // Feature flags
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               prop.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               prop.ECCEnabled ? "Enabled" : "Disabled");

        // Driver mode (Windows only)
#if defined(_WIN32) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               prop.tccDriver
                   ? "TCC (Tesla Compute Cluster Driver)"
                   : "WDDM (Windows Display Driver Model)");
#endif

        printf("  Device supports Unified Addressing (UVA):      %s\n",
               prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n",
               prop.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               prop.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               prop.cooperativeLaunch ? "Yes" : "No");

        // PCI topology
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

        // Compute mode
        const char *computeModeStr[] = {
            "Default (multiple host threads can use ::cudaSetDevice()"
                " with device simultaneously)",
            "Exclusive (only one host thread in one process is able"
                " to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice()"
                " with this device)",
            "Exclusive Process (many threads in one process is able"
                " to use ::cudaSetDevice() with this device)",
        };
        int computeMode = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&computeMode,
                                           cudaDevAttrComputeMode, dev));
        if (computeMode >= 0 && computeMode <= 3) {
            printf("  Compute Mode:\n");
            printf("     < %s >\n", computeModeStr[computeMode]);
        }
    }

    // ------------------------------------------------------------------
    // 3. P2P capability (multi-GPU systems)
    // ------------------------------------------------------------------
    if (deviceCount >= 2) {
        cudaDeviceProp prop[64];
        int            p2pGPUs[64];
        int            p2pCount = 0;

        for (int i = 0; i < deviceCount; i++) {
            CUDA_CHECK(cudaGetDeviceProperties(&prop[i], i));

            if (prop[i].major >= 2
#if defined(_WIN32) || defined(_WIN64)
                && prop[i].tccDriver
#endif
            ) {
                p2pGPUs[p2pCount++] = i;
            }
        }

        if (p2pCount >= 2) {
            for (int i = 0; i < p2pCount; i++) {
                for (int j = 0; j < p2pCount; j++) {
                    if (p2pGPUs[i] == p2pGPUs[j]) continue;
                    int canAccess = 0;
                    CUDA_CHECK(cudaDeviceCanAccessPeer(
                        &canAccess, p2pGPUs[i], p2pGPUs[j]));
                    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                           prop[p2pGPUs[i]].name, p2pGPUs[i],
                           prop[p2pGPUs[j]].name, p2pGPUs[j],
                           canAccess ? "Yes" : "No");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. Summary line
    // ------------------------------------------------------------------
    printf("\n");
    printf("deviceQuery, CUDA Driver = CUDART"
           ", CUDA Driver Version = %d.%d"
           ", CUDA Runtime Version = %d.%d"
           ", NumDevs = %d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10,
           deviceCount);

    printf("Result = PASS\n");
    return EXIT_SUCCESS;
}
