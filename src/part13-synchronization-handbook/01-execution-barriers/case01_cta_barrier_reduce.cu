#include "common.cuh"

/*
 * 目的：
 *   对比纯 CTA 会合与“会合 + predicate reduction”。
 *
 * 验证：
 *   64 个线程写 shared 后求和为 0+...+63=2016；
 *   偶数线程数为 32；all(tid<64)=1；any(tid==63)=1。
 *
 * 观察：
 *   __syncthreads()              → bar.sync / BAR.SYNC
 *   __syncthreads_count(pred)    → bar.red.popc
 *   __syncthreads_and/or(pred)   → bar.red.and/or
  补充说明：PTX指令中有[]，比如	st.shared.u32 	[%r7], %r1;这是指访问memory地址，[]中的%r7是内存地址，%r1是值写到这个地址中。
 */
__global__ void cta_barrier_reduce(int* results) {
  __shared__ int values[64];
  const int tid = threadIdx.x;

  values[tid] = tid;
  __syncthreads();  //ptx:bar.sync 只等待所有线程到达，不执行任何操作。

  if (tid == 0) {  // ptx: setp:set predicate,ne:not equal 相当于tid!=0,.s32表示32位有符号数比较。
    int sum = 0;    // 在ptx中实现的是@%p1 bra 	$L__BB0_2;，意思是%p1为true则跳转到$L__BB0_2标签处执行，只有线程0为false执行这串加法。
    for (int i = 0; i < 64; ++i) {
      sum += values[i];
    }
    results[0] = sum;
  }

  const int even_count = __syncthreads_count((tid & 1) == 0);  //ptx:bar.red.popc ,red是reduce缩写。等待所有线程到达，并统计perdicate为true的线程数。
  const int all_in_range = __syncthreads_and(tid < 64);  //ptx:bar.red.and 等待所有线程到达，并统计所有线程perdicate是否都为true。
  const int any_last = __syncthreads_or(tid == 63);  //ptx:bar.red.or 等待所有线程到达，并统计是否至少有一个线程perdicate为true。

  if (tid == 0) {
    results[1] = even_count;
    results[2] = all_in_range;
    results[3] = any_last;
  }
}

int main() {
  int* device_results = nullptr;
  CUDA_CHECK(cudaMalloc(&device_results, 4 * sizeof(int)));

  cta_barrier_reduce<<<1, 64>>>(device_results);
  finish_kernel();

  int results[4]{};
  CUDA_CHECK(cudaMemcpy(results, device_results, sizeof(results),
                        cudaMemcpyDeviceToHost));
  int status = require_equal("CTA shared sum", results[0], 2016);
  status |= require_equal("bar.red popc", results[1], 32);
  status |= require_equal("bar.red and", results[2], 1);
  status |= require_equal("bar.red or", results[3], 1);

  CUDA_CHECK(cudaFree(device_results));
  return status;
}
