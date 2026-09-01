#include "common.cuh"

#include <cooperative_groups.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

/*
 * 目的：
 *   用 cuda::pipeline 表达 producer acquire/commit 与 consumer wait/release，
 *   观察 Ampere+ 是否生成 cp.async + commit_group/wait_group。
 *
 * 验证：
 *   128 个 int 经 global→shared 异步 copy 后各加 1，输出应为 input+1。
 */
__global__ void cp_async_pipeline_case(const int* input, int* output) {
  constexpr int count = 128;
  constexpr int stages = 1;
  __shared__ alignas(16) int tile[count];
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope_block, stages> pipeline_state;

  cg::thread_block block = cg::this_thread_block();
  auto pipeline = cuda::make_pipeline(block, &pipeline_state);

  pipeline.producer_acquire();
  cuda::memcpy_async(block, tile, input,
                     cuda::aligned_size_t<16>(count * sizeof(int)),
                     pipeline);
  pipeline.producer_commit();

  pipeline.consumer_wait();
  output[threadIdx.x] = tile[threadIdx.x] + 1;
  pipeline.consumer_release();
}

int main() {
  constexpr int count = 128;
  int host_input[count];
  for (int i = 0; i < count; ++i) {
    host_input[i] = i;
  }

  int* input = nullptr;
  int* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, sizeof(host_input)));
  CUDA_CHECK(cudaMalloc(&output, sizeof(host_input)));
  CUDA_CHECK(cudaMemcpy(input, host_input, sizeof(host_input),
                        cudaMemcpyHostToDevice));

  cp_async_pipeline_case<<<1, count>>>(input, output);
  finish_kernel();

  int host_output[count]{};
  CUDA_CHECK(cudaMemcpy(host_output, output, sizeof(host_output),
                        cudaMemcpyDeviceToHost));
  int status = EXIT_SUCCESS;
  for (int i = 0; i < count; ++i) {
    if (host_output[i] != i + 1) {
      std::cerr << "mismatch at " << i << '\n';
      status = EXIT_FAILURE;
      break;
    }
  }
  std::cout << "cp.async pipeline: "
            << (status == EXIT_SUCCESS ? "passed" : "failed") << '\n';

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  return status;
}
