#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 03：plain、relaxed atomic 与 acquire atomic load
 *
 * 测试目的：
 *   区分三个逐步增强但含义不同的概念：
 *   普通读取、只保证 atomicity 的 relaxed 读取、可用于消费发布值的 acquire。
 *
 * 运行时验证：
 *   flag 预置为 7，三种 load 都应得到 7。数值相同是预期结果；本 case 真正
 *   要比较的是生成指令，而不是用单线程顺序证明 acquire 的并发可见性。
 *
 * PTX/SASS 观察：
 *   plain：   ld.global
 *   relaxed： ld.relaxed.gpu
 *   acquire： ld.acquire.gpu
 *   acquire 在 SASS 中可能对应 STRONG load、cache invalidation 或 fence 序列。
 *
 * 语义结论：
 *   acquire 只有读取到匹配 release 发布的值后，才能让其后的 payload 读取
 *   依赖该发布；它本身不会凭空同步数据。
 */
extern "C" __global__ void case03_plain_load(const int* flag, int* observed) {
  *observed = *flag;
}

extern "C" __global__ void case03_relaxed_load(int* flag, int* observed) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  *observed = atomic_flag.load(cuda::memory_order_relaxed);
}

extern "C" __global__ void case03_acquire_load(int* flag, int* observed) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  *observed = atomic_flag.load(cuda::memory_order_acquire);
}

int main() {
  int* flag = make_device_int(7);
  int* observed = make_device_int();
  int status = EXIT_SUCCESS;

  case03_plain_load<<<1, 1>>>(flag, observed);
  finish_kernel();
  status |= require_equal("plain load", read_device_int(observed), 7);

  case03_relaxed_load<<<1, 1>>>(flag, observed);
  finish_kernel();
  status |= require_equal("relaxed atomic load", read_device_int(observed), 7);

  case03_acquire_load<<<1, 1>>>(flag, observed);
  finish_kernel();
  status |= require_equal("acquire atomic load", read_device_int(observed), 7);

  CUDA_CHECK(cudaFree(observed));
  CUDA_CHECK(cudaFree(flag));
  return status;
}
