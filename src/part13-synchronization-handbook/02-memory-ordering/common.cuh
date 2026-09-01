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

inline int* make_device_int(int initial_value = 0) {
  int* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(pointer, &initial_value, sizeof(int),
                        cudaMemcpyHostToDevice));
  return pointer;
}

inline int read_device_int(const int* pointer) {
  int value = 0;
  CUDA_CHECK(cudaMemcpy(&value, pointer, sizeof(int), cudaMemcpyDeviceToHost));
  return value;
}

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
