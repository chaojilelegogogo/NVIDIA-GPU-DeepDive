#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 02：relaxed atomic RMW 与 release atomic RMW
 *
 * 测试目的：
 *   保持 fetch_add、device scope 和操作数完全相同，只改变 memory order，
 *   观察“原子性”与“发布语义”在 PTX/SASS 中如何分离。
 *
 * 运行时验证：
 *   counter 初值为 10。relaxed fetch_add 返回 10，release fetch_add 返回 11，
 *   最终 counter 为 12。两种 order 都必须保证 fetch_add 本身的原子性。
 *
 * PTX/SASS 观察：
 *   relaxed：atom.add.relaxed.gpu
 *   release：atom.add.release.gpu
 *   SASS 中两者可能使用同一 ATOM 主体，而 release 额外出现 MEMBAR，或由
 *   指令的 ordering/STRONG 变体表达。
 *
 * 语义结论：
 *   relaxed 不发布其它 payload；release 还保证此前访问不能越过该 RMW。
 */
extern "C" __global__ void case02_relaxed_add(int* counter,
                                               int* old_value) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_counter(*counter);
  *old_value =
      atomic_counter.fetch_add(1, cuda::memory_order_relaxed);
}

extern "C" __global__ void case02_release_add(int* counter,
                                               int* old_value) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_counter(*counter);
  // Keeping the returned value observable encourages an atom.* RMW rather than
  // a reduction-only lowering, making the PTX comparison easier to read.
  *old_value =
      atomic_counter.fetch_add(1, cuda::memory_order_release);
}

int main() {
  int* counter = make_device_int(10);
  int* old_value = make_device_int();

  case02_relaxed_add<<<1, 1>>>(counter, old_value);
  finish_kernel();
  int status =
      require_equal("relaxed old value", read_device_int(old_value), 10);

  case02_release_add<<<1, 1>>>(counter, old_value);
  finish_kernel();
  status |=
      require_equal("release old value", read_device_int(old_value), 11);
  status |= require_equal("final counter", read_device_int(counter), 12);

  CUDA_CHECK(cudaFree(old_value));
  CUDA_CHECK(cudaFree(counter));
  return status;
}
