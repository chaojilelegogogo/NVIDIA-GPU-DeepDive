#include "common.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/*
 * 目的：
 *   验证 Hopper+ cluster launch、DSM 地址映射和 cluster execution barrier。
 *
 * 验证：
 *   rank 1 CTA 在自己的 shared memory 写入 11；cluster.sync 后 rank 0
 *   通过 map_shared_rank 读取该 remote shared value，结果应为 11。
 *
 * 边界：
 *   需要 compute capability 9.0+ 和 cluster launch；不支持时测试安全跳过。
 *
 * 知识点：
 *   - extern __shared__ 声明动态share mem
 *   - cg::cluster_group 返回当前线程所在thread block cluster信息，比如这个cluster有几个CTA等
 *   - cg::cluster_group.block_rank() 返回当前线程所在thread block 在cluster中的rank
 *   - cg::cluster_group.sync() 同步当前cluster所有CTA
 *   - cg::cluster_group.map_shared_rank(local_shared, 1) 将local_shared 映射到 remote shared memory
 *   - ptx中的.reg .b32 	%r<7>;表示32位寄存器有7个，编号是0-6..reg .b64 	%rd<6>;表示64位寄存器有6个。但是这里应该都是虚拟寄存器。
 *   - 似乎ptx代码里判断都是走不成立继续，比如@%p1 bra 	$L__BB0_2;判断谓词寄存器p1为true就走BB0_2，否则继续执行下一条指令。
 *   - 比较的时候通常是把等于转换为不等于，setp.ne.s32 	%p1, %r2, 0;这里的ne就是not equal。
 */
__global__ void cluster_sync_case(int* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // 对应PTX：.extern .shared .align 16 .b8 local_shared[];  没有指定大小，因为编译期并不知道大小，运行时由dirver分配
  extern __shared__ int local_shared[]; // extern __shared__ 声明动态share mem  
  cg::cluster_group cluster = cg::this_cluster();  // 返回当前线程所在thread block cluster信息，比如这个cluster有几个CTA等
  // 返回当前线程所在thread block 在cluster中的rank。从hopper开始新增%cluster_ctarank这个特殊寄存器，对应cluster.block_rank()
  // PTX中，以%开头的就是寄存器标识，包括普通寄存器和特殊寄存器。blockIdx.x（ptx：%ctaid.x）等都是特殊寄存器。
  const unsigned rank = cluster.block_rank();  // ptx：mov.u32 	%r1, %cluster_ctarank;
  if (threadIdx.x == 0) {
    local_shared[0] = static_cast<int>(rank) + 10;
  }
  
  // 同步当前cluster所有CTA。
  // 对应PTX：barrier.cluster.arrive;和barrier.cluster.wait;表示到达和等待其他的。
  cluster.sync();

  // 这里把&&这个条件转换为||   	or.b32  	%r4, %r1, %r2;	setp.ne.s32 	%p2, %r4, 0;
  if (rank == 0 && threadIdx.x == 0) {
    // 输入当前rank0 的 share mem的地址，获取cta 1的地址
    // 	ptx：mapa.u64	%rd4, %rd3, 1;
    int* remote = cluster.map_shared_rank(local_shared, 1);
    *output = remote[0];
  }

  // PTX:	barrier.cluster.arrive;  	barrier.cluster.wait;
  cluster.sync();
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *output = -1;
  }
#endif
}

int main() {
  int cluster_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &cluster_supported, cudaDevAttrClusterLaunch, 0));
  if (!cluster_supported) {
    std::cout << "cluster launch unsupported: skipped\n";
    return EXIT_SUCCESS;
  }

  int* output = nullptr;
  CUDA_CHECK(cudaMalloc(&output, sizeof(int)));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(2, 1, 1);
  config.blockDim = dim3(32, 1, 1);
  config.dynamicSmemBytes = sizeof(int);

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = 2;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, cluster_sync_case, output));
  finish_kernel();

  int observed = 0;
  CUDA_CHECK(cudaMemcpy(&observed, output, sizeof(int),
                        cudaMemcpyDeviceToHost));
  const int status = require_equal("remote DSM value", observed, 11);
  CUDA_CHECK(cudaFree(output));
  return status;
}
