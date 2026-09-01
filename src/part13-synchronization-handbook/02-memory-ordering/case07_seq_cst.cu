#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 07：relaxed/release 与 sequential consistency
 *
 * 测试目的：
 *   观察 CUDA C++ memory_order_seq_cst 通常不是一条名为 seq_cst 的
 *   load/store，而是由 fence.sc 与较弱访问组合实现。
 *
 * 运行时验证：
 *   默认 stream 中依次写入 1、2、3，再执行 seq_cst load；最终应读取到 3。
 *   这个结果主要验证调用与输出，不证明多个并发线程之间的 SC litmus test。
 *
 * PTX/SASS 观察：
 *   relaxed store：st.relaxed.gpu
 *   release store：st.release.gpu
 *   seq_cst store：常见 fence.sc.gpu + st.relaxed.gpu
 *   seq_cst load： 常见 fence.sc.gpu + ld.acquire.gpu
 *   SASS 中重点比较 MEMBAR.SC.GPU、MEMBAR.ALL.GPU 与 STRONG load/store。
 *
 * 语义结论：
 *   seq_cst 在 acquire/release 基础上，还让相关 seq_cst 操作参与一致全序；
 *   它不是 CTA/grid execution barrier，也不意味着所有线程停在同一点。
 */
extern "C" __global__ void case07_relaxed_store(int* flag) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  atomic_flag.store(1, cuda::memory_order_relaxed);
}

extern "C" __global__ void case07_release_store(int* flag) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  atomic_flag.store(2, cuda::memory_order_release);
}

extern "C" __global__ void case07_seq_cst_store(int* flag) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  atomic_flag.store(3, cuda::memory_order_seq_cst);
}

extern "C" __global__ void case07_seq_cst_load(int* flag, int* observed) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  *observed = atomic_flag.load(cuda::memory_order_seq_cst);
}

int main() {
  int* flag = make_device_int();
  int* observed = make_device_int();

  case07_relaxed_store<<<1, 1>>>(flag);
  case07_release_store<<<1, 1>>>(flag);
  case07_seq_cst_store<<<1, 1>>>(flag);
  case07_seq_cst_load<<<1, 1>>>(flag, observed);
  finish_kernel();

  int status =
      require_equal("seq_cst observed value", read_device_int(observed), 3);

  CUDA_CHECK(cudaFree(observed));
  CUDA_CHECK(cudaFree(flag));
  return status;
}
