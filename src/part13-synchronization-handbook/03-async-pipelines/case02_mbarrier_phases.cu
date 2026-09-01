#include "common.cuh"

#include <cuda/barrier>

/*
 * 目的：
 *   将 barrier 当成可复用 phase 对象，而不是一次性的 __syncthreads。
 *
 * 验证：
 *   两轮中每个线程写 shared，再保存 arrival token 并等待当前 phase；
 *   lane 0 分别读取 lane 31 写入的 31 和 131。
 *
 * 观察：
 *   cuda::barrier 在支持的目标上通常降低为 mbarrier init/arrive/try_wait，
 *   token/parity 用于区分每一代 phase。
 */
__global__ void mbarrier_phases_case(int* results) {
  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier_t barrier;
  __shared__ int values[32];

  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  __syncthreads();

  for (int iteration = 0; iteration < 2; ++iteration) {
    values[threadIdx.x] = iteration * 100 + threadIdx.x;
    auto ready = barrier.arrive();
    barrier.wait(static_cast<barrier_t::arrival_token&&>(ready));

    if (threadIdx.x == 0) {
      results[iteration] = values[31];
    }

    // 第二个 phase 保证所有读取完成后才允许下一轮覆盖 values。
    auto consumed = barrier.arrive();
    barrier.wait(static_cast<barrier_t::arrival_token&&>(consumed));
  }

  if (threadIdx.x == 0) {
    (&barrier)->~barrier_t();
  }
}

int main() {
  int* results = nullptr;
  CUDA_CHECK(cudaMalloc(&results, 2 * sizeof(int)));

  mbarrier_phases_case<<<1, 32>>>(results);
  finish_kernel();

  int host_results[2]{};
  CUDA_CHECK(cudaMemcpy(host_results, results, sizeof(host_results),
                        cudaMemcpyDeviceToHost));
  int status = require_equal("phase 0", host_results[0], 31);
  status |= require_equal("phase 2", host_results[1], 131);

  CUDA_CHECK(cudaFree(results));
  return status;
}
