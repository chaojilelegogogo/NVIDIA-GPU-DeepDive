#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error__ = (call);                                            \
    if (error__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "  \
                << cudaGetErrorString(error__) << '\n';                      \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (false)

inline void finish_kernel() {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
