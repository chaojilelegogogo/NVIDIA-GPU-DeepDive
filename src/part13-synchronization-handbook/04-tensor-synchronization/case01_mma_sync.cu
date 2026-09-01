#include "common.cuh"

#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

/*
 * 目的：
 *   建立同步式 Tensor Core 基线：一个 warp 集体执行 mma_sync，结果通过
 *   普通寄存器数据依赖可用，不需要 WGMMA/tcgen05 的 async wait group。
 *
 * 验证：
 *   A、B 都是 16x16 全 1 矩阵，C=A*B，因此每个输出元素应为 16。
 *
 * 观察：
 *   WMMA API 通常降低为 load/`mma.sync`/store，SASS 常见 HMMA。
 */
__global__ void mma_sync_case(const half* matrix_a,
                              const half* matrix_b,
                              float* matrix_c) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16,
                 half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16,
                 half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

  wmma::load_matrix_sync(a, matrix_a, 16);
  wmma::load_matrix_sync(b, matrix_b, 16);
  wmma::fill_fragment(c, 0.0f);
  wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(matrix_c, c, 16, wmma::mem_row_major);
}

int main() {
  constexpr int elements = 16 * 16;
  half host_a[elements];
  half host_b[elements];
  for (int i = 0; i < elements; ++i) {
    host_a[i] = __float2half(1.0f);
    host_b[i] = __float2half(1.0f);
  }

  half* matrix_a = nullptr;
  half* matrix_b = nullptr;
  float* matrix_c = nullptr;
  CUDA_CHECK(cudaMalloc(&matrix_a, sizeof(host_a)));
  CUDA_CHECK(cudaMalloc(&matrix_b, sizeof(host_b)));
  CUDA_CHECK(cudaMalloc(&matrix_c, elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(matrix_a, host_a, sizeof(host_a),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(matrix_b, host_b, sizeof(host_b),
                        cudaMemcpyHostToDevice));

  mma_sync_case<<<1, 32>>>(matrix_a, matrix_b, matrix_c);
  finish_kernel();

  float host_c[elements]{};
  CUDA_CHECK(cudaMemcpy(host_c, matrix_c, sizeof(host_c),
                        cudaMemcpyDeviceToHost));
  int status = EXIT_SUCCESS;
  for (float value : host_c) {
    if (value != 16.0f) {
      status = EXIT_FAILURE;
      break;
    }
  }
  std::cout << "mma.sync result: "
            << (status == EXIT_SUCCESS ? "passed" : "failed") << '\n';

  CUDA_CHECK(cudaFree(matrix_c));
  CUDA_CHECK(cudaFree(matrix_b));
  CUDA_CHECK(cudaFree(matrix_a));
  return status;
}
