cmake_minimum_required(VERSION 3.24)

# "native" is convenient on a development GPU. For reproducible builds use,
# for example: -DNVGPU_CUDA_ARCH=sm_87 or -DNVGPU_CUDA_ARCH=sm_90a.
set(NVGPU_CUDA_ARCH "native" CACHE STRING
    "nvcc architecture used for PTX/CUBIN inspection artifacts")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(NVGPU_CUDA_ARCH STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    string(REGEX REPLACE "^sm_" "" _cmake_cuda_arch "${NVGPU_CUDA_ARCH}")
    # CMake does not understand the architecture-specific 'a' suffix.
    string(REGEX REPLACE "a$" "" _cmake_cuda_arch "${_cmake_cuda_arch}")
    set(CMAKE_CUDA_ARCHITECTURES "${_cmake_cuda_arch}")
  endif()
endif()

project(nvgpu_learn_cases LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_testing()
add_subdirectory(src)
