#include "common.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/*
 * 目的：
 *   区分普通 kernel block 调度与 cooperative grid execution barrier。
 *
 * 验证：
 *   block 0 写 payload=77；整个 cooperative grid 执行 grid.sync；
 *   block 1 随后读取 payload，输出应为 77。
 *
 * 边界：
 *   必须使用 cudaLaunchCooperativeKernel；设备不支持 cooperative launch 时跳过。
 */
__global__ void grid_sync_case(int* payload, int* observed) {
  // ptx出现mov.u32 %r4, %envreg2;，应该是通过%envreg2判断当前是否支持grid.sync
  // 这里有点复杂，%envreg2和%envreg1分别表示高32位和低32位，通过or.b64  	%rd2, %rd9, %rd7; 合并成一个64位寄存器数据。
  cg::grid_group grid = cg::this_grid();  

  // 特殊寄存器：	mov.u32 	%r1, %ctaid.x;  mov.u32 	%r2, %tid.x;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *payload = 77;
  }
    // 然后通过setp.ne.b64 %p1, %rd2, 0; 判断是否支持grid.sync。不支持则直接trap;，也就是崩掉。那说明这段代码如果
    // 没有通过cudaLaunchCooperativeKernel发射，在编译时不会报错，但运行时会崩。
    // 真正进入grid.sync时，先barrier.sync 	0; 它是CTA barrier，也就是__syncthreads()。有这步的原因是后面的线程需要选一个代表线程做atomic。
    // 然后根据ctaid.x y z计算block的总数量
    // setp.eq.s32 	%p4, %r20, %r22;这之前的几行就是选出0线程，执行后面的atomicadd操作。selp是 predicate 条件选择指令，类似于三目运算符。
    // selp.b32 	%r13, %r23, 1, %p4;这里判断是不是第0个block，如果是就给 atomic counter 一个特殊增量，用于初始化 barrier epoch。如果不是普通 block 增加 1。
    // 这里设计block0为leader block，为什么要有leader block，这里也很复杂，暂时没有搞明白。
    // atom.add.release.gpu.u32 %r12,[%rd10],%r13; scope为gpu，mem的order是release，执行atomicadd操作进行计数。
    // 由于这里是atom.add.release，在release之前所有memory操作要完成，这样就能保证后面payload结果是可见的。
    // ld.acquire.gpu.u32 %r24,[%rd10]; 循环里面不断获取counter，直到满足条件执行后面的。这里用了acquire，保证后面的memory操作不会先被执行。
  grid.sync();

  if (blockIdx.x == 1 && threadIdx.x == 0) {
    *observed = *payload;
  }
}

int main() {
  int cooperative_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &cooperative_supported, cudaDevAttrCooperativeLaunch, 0));
  if (!cooperative_supported) {
    std::cout << "cooperative launch unsupported: skipped\n";
    return EXIT_SUCCESS;
  }

  int* payload = nullptr;
  int* observed = nullptr;
  CUDA_CHECK(cudaMalloc(&payload, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&observed, sizeof(int)));
  CUDA_CHECK(cudaMemset(payload, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(observed, 0, sizeof(int)));

  void* args[] = {&payload, &observed};
  CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(grid_sync_case), dim3(2), dim3(32), args));
  finish_kernel();

  int host_observed = 0;
  CUDA_CHECK(cudaMemcpy(&host_observed, observed, sizeof(int),
                        cudaMemcpyDeviceToHost));
  const int status = require_equal("cooperative grid value",
                                   host_observed, 77);

  CUDA_CHECK(cudaFree(observed));
  CUDA_CHECK(cudaFree(payload));
  return status;
}
