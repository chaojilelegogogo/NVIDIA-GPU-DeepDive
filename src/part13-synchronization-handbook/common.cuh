#pragma once

#include <cuda_runtime.h>

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

inline int require_equal(const char* label, int actual, int expected) {
  if (actual != expected) {
    std::cerr << label << ": expected " << expected << ", got " << actual
              << '\n';
    return EXIT_FAILURE;
  }
  std::cout << label << ": " << actual << '\n';
  return EXIT_SUCCESS;
}
