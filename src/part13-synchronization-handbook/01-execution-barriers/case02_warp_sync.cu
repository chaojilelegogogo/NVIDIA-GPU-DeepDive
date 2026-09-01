#include "common.cuh"

/*
 * 目的：
 *   验证 Volta+ warp 内 shared-memory handoff 需要显式参与 mask。
 *
 * 验证：
 *   每个 lane 写入 shared[lane]，__syncwarp 后读取前一 lane；
 *   lane 0 环绕读取 lane 31，因此输出应为 31。
 *
 * 观察：
 *   __syncwarp(0xffffffff) → bar.warp.sync → 常见 SASS WARPSYNC。
 */
__global__ void warp_sync_case(int* output) {
  __shared__ int values[32];
  const int lane = threadIdx.x & 31;
  constexpr unsigned full_mask = 0xffffffffu;  // 表示需要同步的线程掩码。

  values[lane] = lane;
  // ptx: bar.warp.sync 	-1; -1应该是0xffffffffu的十进制表示？
  __syncwarp(full_mask);  // 从volta开始引入independent thread scheduling，warp内线程可以不同步推进。

  const int previous = values[(lane + 31) & 31];
  if (lane == 0) {
    *output = previous;
  }
}

int main() {
  int* output = nullptr;
  CUDA_CHECK(cudaMalloc(&output, sizeof(int)));

  warp_sync_case<<<1, 32>>>(output);
  finish_kernel();

  int observed = 0;
  CUDA_CHECK(cudaMemcpy(&observed, output, sizeof(int),
                        cudaMemcpyDeviceToHost));
  const int status = require_equal("warp handoff", observed, 31);
  CUDA_CHECK(cudaFree(output));
  return status;
}
