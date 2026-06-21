cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(cuda_rsp LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER "D:/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── elementwise_add ──
add_executable(elementwise_add
    easy_examples/elementwise_add/main_elementwise_add.cu
    src/elementwise_add.cu
)

# ── histogram ──
add_executable(histogram
    easy_examples/histogram/main_histogram.cu
    src/histogram.cu
)

# ── sigmoid ──
add_executable(sigmoid
    easy_examples/sigmoid/main_sigmoid.cu
    src/sigmoid.cu
)

# ── relu ──
add_executable(relu
    easy_examples/relu/main_relu.cu
    src/relu.cu
)


