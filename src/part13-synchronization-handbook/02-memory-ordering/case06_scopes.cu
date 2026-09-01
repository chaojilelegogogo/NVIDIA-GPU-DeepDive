#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 06：同一 relaxed atomic 在不同 thread scope 下的变化
 *
 * 测试目的：
 *   固定 fetch_add 与 relaxed order，只改变 block/cluster/device/system scope，
 *   观察 scope 如何映射到 PTX 和 SASS 的可观察范围。
 *
 * 运行时验证：
 *   value 初值为 0，各 kernel 依次执行；scope 不改变单线程算术结果。
 *   cluster scope 仅在 compute capability 9.0+ 执行。
 *
 * PTX/SASS 观察：
 *   block  → atom.add.relaxed.cta   → 常见 SASS scope SM
 *   cluster→ atom.add.relaxed.cluster（CUDA 13.0 libcu++ 尚未导出
 *            thread_scope_cluster，本 case 用内联 PTX 观察）
 *   device → atom.add.relaxed.gpu   → 常见 SASS scope GPU
 *   system → atom.add.relaxed.sys   → 常见 SASS scope SYS
 *
 * 语义结论：
 *   scope 回答哪些观察者可依赖该操作，不代表地址空间，也不会让参与线程
 *   会合。scope 小于真实参与范围会出错；盲目使用更大 scope 可能增加成本。
 */
extern "C" __global__ void case06_block_scope(int* value, int* old_value) {
  cuda::atomic_ref<int, cuda::thread_scope_block> atomic_value(*value);
  *old_value = atomic_value.fetch_add(1, cuda::memory_order_relaxed);
}

extern "C" __global__ void case06_cluster_scope(int* value, int* old_value) {
  // CUDA 13.0 libcu++ 尚未导出 cuda::thread_scope_cluster 枚举；用 PTX 直接观察 .cluster。
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  unsigned int old = 0;
  asm volatile("atom.add.relaxed.cluster.u32 %0, [%1], 1;"
               : "=r"(old)
               : "l"(value)
               : "memory");
  *old_value = static_cast<int>(old);
#else
  (void)value;
  (void)old_value;
#endif
}

extern "C" __global__ void case06_device_scope(int* value, int* old_value) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_value(*value);
  *old_value = atomic_value.fetch_add(1, cuda::memory_order_relaxed);
}

extern "C" __global__ void case06_system_scope(int* value, int* old_value) {
  cuda::atomic_ref<int, cuda::thread_scope_system> atomic_value(*value);
  *old_value = atomic_value.fetch_add(1, cuda::memory_order_relaxed);
}

int main() {
  int* value = make_device_int();
  int* old_value = make_device_int();
  int status = EXIT_SUCCESS;

  case06_block_scope<<<1, 1>>>(value, old_value);
  finish_kernel();
  status |= require_equal("block-scope old value",
                          read_device_int(old_value), 0);

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  int next_expected = 1;
  if (properties.major >= 9) {
    case06_cluster_scope<<<1, 1>>>(value, old_value);
    finish_kernel();
    status |= require_equal("cluster-scope old value",
                            read_device_int(old_value), next_expected++);
  } else {
    std::cout << "cluster scope requires compute capability 9.0+: skipped\n";
  }

  case06_device_scope<<<1, 1>>>(value, old_value);
  finish_kernel();
  status |= require_equal("device-scope old value",
                          read_device_int(old_value), next_expected++);

  case06_system_scope<<<1, 1>>>(value, old_value);
  finish_kernel();
  status |= require_equal("system-scope old value",
                          read_device_int(old_value), next_expected++);
  status |= require_equal("final value", read_device_int(value), next_expected);

  CUDA_CHECK(cudaFree(old_value));
  CUDA_CHECK(cudaFree(value));
  return status;
}
