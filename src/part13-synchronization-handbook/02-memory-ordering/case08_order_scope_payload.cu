#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 08：Order × Scope 的最小 payload 发布协议
 *
 * 测试目的：
 *   把文档 13.2.3 的三问落到一个 CTA 内协议：
 *   1. Order：producer 先写 payload，再用 release 发布 flag；
 *   2. Scope：两端都使用 thread_scope_block（.cta）；
 *   3. Proxy：payload/flag 都走普通 generic global/shared 路径。
 *
 * 运行时验证：
 *   tid0 写入 payload=42 并以 release 置 flag=1；
 *   其它线程 acquire-load flag 成功后读取 payload，最终 observed=42。
 *
 * PTX/SASS 观察：
 *   producer：普通 st.global（payload）+ st.release.cta（flag）
 *   consumer：ld.acquire.cta（flag）+ 普通 ld.global（payload）
 *
 * 语义边界：
 *   本 case 故意不用 __syncthreads()，以便观察 release/acquire 自己完成
 *   CTA 内发布。跨 CTA 时必须把 scope 升到 device/cluster，并处理驻留问题。
 */
__global__ void case08_cta_payload_publish(int* payload, int* flag,
                                           int* observed) {
  cuda::atomic_ref<int, cuda::thread_scope_block> atomic_flag(*flag);

  if (threadIdx.x == 0) {
    *payload = 42;
    atomic_flag.store(1, cuda::memory_order_release);
  } else {
    while (atomic_flag.load(cuda::memory_order_acquire) == 0) {
    }
    observed[threadIdx.x] = *payload;
  }
}

int main() {
  constexpr int kThreads = 32;
  int* payload = make_device_int();
  int* flag = make_device_int();
  int* observed = nullptr;
  CUDA_CHECK(cudaMalloc(&observed, kThreads * sizeof(int)));
  CUDA_CHECK(cudaMemset(observed, 0, kThreads * sizeof(int)));

  case08_cta_payload_publish<<<1, kThreads>>>(payload, flag, observed);
  finish_kernel();

  int host_observed[kThreads]{};
  CUDA_CHECK(cudaMemcpy(host_observed, observed, sizeof(host_observed),
                        cudaMemcpyDeviceToHost));

  int status = require_equal("payload", read_device_int(payload), 42);
  status |= require_equal("flag", read_device_int(flag), 1);
  for (int i = 1; i < kThreads; ++i) {
    if (host_observed[i] != 42) {
      std::cerr << "consumer lane " << i << " observed " << host_observed[i]
                << ", expected 42\n";
      status = EXIT_FAILURE;
      break;
    }
  }
  if (status == EXIT_SUCCESS) {
    std::cout << "cta payload publish: all consumers saw 42\n";
  }

  CUDA_CHECK(cudaFree(observed));
  CUDA_CHECK(cudaFree(flag));
  CUDA_CHECK(cudaFree(payload));
  return status;
}
