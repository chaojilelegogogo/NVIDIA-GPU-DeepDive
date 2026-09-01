#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 05：显式 __threadfence() 与 release store
 *
 * 测试目的：
 *   比较“普通 payload store + 显式 device fence + 普通 flag store”和
 *   “普通 payload store + release atomic flag store”的编译方式。
 *
 * 运行时验证：
 *   threadfence 版本应写出 data=100、flag=1；
 *   release-store 版本应写出 data=200、flag=1。
 *
 * PTX/SASS 观察：
 *   threadfence：st.global → membar.gl/fence.sc.gpu → st.global
 *   release：    st.global → st.release.gpu
 *   对比显式 fence 与附着在发布操作上的 order 如何被 ptxas 降低。
 *
 * 语义结论：
 *   __threadfence() 只排序 producer 的访问，不会让 consumer 等待。第一种
 *   写法中的普通 flag 若被另一线程并发读取会产生数据竞争，不能独立构成
 *   完整协议；实际发布应使用 atomic release 与匹配的 atomic acquire。
 */
extern "C" __global__ void case05_threadfence_publish(int* data, int* flag) {
  *data = 100;
  __threadfence();
  *flag = 1;
}

extern "C" __global__ void case05_release_publish(int* data, int* flag) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  *data = 200;
  atomic_flag.store(1, cuda::memory_order_release);
}

int main() {
  int* data = make_device_int();
  int* flag = make_device_int();

  case05_threadfence_publish<<<1, 1>>>(data, flag);
  finish_kernel();
  int status = require_equal("threadfence payload", read_device_int(data), 100);
  status |= require_equal("threadfence flag", read_device_int(flag), 1);

  int zero = 0;
  CUDA_CHECK(cudaMemcpy(flag, &zero, sizeof(int), cudaMemcpyHostToDevice));
  case05_release_publish<<<1, 1>>>(data, flag);
  finish_kernel();
  status |= require_equal("release payload", read_device_int(data), 200);
  status |= require_equal("release flag", read_device_int(flag), 1);

  CUDA_CHECK(cudaFree(flag));
  CUDA_CHECK(cudaFree(data));
  return status;
}
