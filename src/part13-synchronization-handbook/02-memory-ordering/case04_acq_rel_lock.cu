#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 04：acq_rel exchange 实现最小锁协议
 *
 * 测试目的：
 *   观察 atomic exchange 同时携带 acquire 与 release 时的 PTX/SASS，并理解
 *   锁获取、临界区和 release 解锁之间的顺序。
 *
 * 运行时验证：
 *   lock 初值为 0，单线程 exchange 应成功取得锁；payload 从 41 增至 42；
 *   release store 解锁后 lock 应恢复为 0。
 *
 * PTX/SASS 观察：
 *   获取：atom.exch.acq_rel.gpu
 *   解锁：st.release.gpu
 *   SASS 常见 ATOM.E.EXCH 与其前后所需的 MEMBAR/STRONG ordering 序列。
 *
 * 语义结论：
 *   acq_rel 的 release 部分约束 exchange 之前的访问，acquire 部分约束
 *   exchange 之后的访问。它不是“操作前 acquire、操作后 release”。
 *
 * 实验边界：
 *   为避免 GPU 自旋锁的进度与 warp divergence 问题，运行时只启动一个线程；
 *   该程序用于观察指令，不是生产级高竞争锁实现。
 */
extern "C" __global__ void case04_acq_rel_lock(int* lock, int* payload) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_lock(*lock);

  while (atomic_lock.exchange(1, cuda::memory_order_acq_rel) == 1) {
    // The example is intentionally minimal. Production GPU locks should use
    // backoff and must account for warp-level progress and contention.
  }

  *payload += 1;
  atomic_lock.store(0, cuda::memory_order_release);
}

int main() {
  int* lock = make_device_int();
  int* payload = make_device_int(41);

  // One thread keeps this executable deterministic. The generated exchange is
  // the same instruction of interest as in a contended launch.
  case04_acq_rel_lock<<<1, 1>>>(lock, payload);
  finish_kernel();

  int status = require_equal("protected payload", read_device_int(payload), 42);
  status |= require_equal("released lock", read_device_int(lock), 0);

  CUDA_CHECK(cudaFree(payload));
  CUDA_CHECK(cudaFree(lock));
  return status;
}
