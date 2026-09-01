#include "common.cuh"

/*
 * 目的：
 *   演示 CTA named barrier 的 arrive（不阻塞）与 sync（等待）配对。
 *
 * 协议：
 *   64 线程、barrier id=1、count=64。
 *   warp0：写 shared payload 后 bar.arrive（登记到达，继续执行）
 *   warp1：bar.sync（等到全部到达）后再读 payload
 *
 * 验证：
 *   waiting warp 读到 payload=42。
 *
 * 观察：
 *   bar.arrive / bar.sync（或目标 Toolkit 的等价 SASS）
 *
 * 边界：
 *   arrive/sync 的参与 count 必须与实际到达者一致；部分 arrive 形式要求
 *   warp-aligned participation。高层 split-phase 更常用 cuda::barrier。
 */
__global__ void bar_arrive_handoff(int* out) {
  __shared__ int payload;
  if (threadIdx.x == 0) {
    payload = 0;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    if (threadIdx.x == 0) {
      payload = 42;
    }
    asm volatile("bar.arrive 1, 64;" ::: "memory");
  } else {
    asm volatile("bar.sync 1, 64;" ::: "memory");
    if (threadIdx.x == 32) {
      out[0] = payload;
    }
  }
}

int main() {
  int* device_out = nullptr;
  CUDA_CHECK(cudaMalloc(&device_out, sizeof(int)));
  CUDA_CHECK(cudaMemset(device_out, 0xff, sizeof(int)));

  bar_arrive_handoff<<<1, 64>>>(device_out);
  finish_kernel();

  int observed = 0;
  CUDA_CHECK(cudaMemcpy(&observed, device_out, sizeof(int),
                        cudaMemcpyDeviceToHost));
  const int status = require_equal("bar.arrive handoff", observed, 42);
  CUDA_CHECK(cudaFree(device_out));
  return status;
}
