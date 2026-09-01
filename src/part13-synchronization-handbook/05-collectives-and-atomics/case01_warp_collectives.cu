#include "common.cuh"

#include <cstdint>

/*
 * 目的：
 *   在同一 warp 中并排观察 activemask、shuffle、vote/ballot 和 match。
 *
 * 验证（lane 0）：
 *   full mask=0xffffffff；shfl_down 得到 lane1 的值2；any(lane==7)=1；
 *   ballot(lane<16)=0x0000ffff；match(lane%4==0)=0x11111111。
 *
 * 观察：
 *   activemask、shfl.sync、vote.sync、match.sync 及对应 SASS。
 */
__global__ void warp_collectives_case(uint32_t* results) {
  const int lane = threadIdx.x & 31;
  const unsigned mask = __activemask();
  const int value = lane + 1;

  const int shuffled = __shfl_down_sync(mask, value, 1);
  const int any_lane_7 = __any_sync(mask, lane == 7);
  const unsigned lower_half = __ballot_sync(mask, lane < 16);
  const unsigned peers = __match_any_sync(mask, lane & 3);

  if (lane == 0) {
    results[0] = mask;
    results[1] = static_cast<unsigned>(shuffled);
    results[2] = static_cast<unsigned>(any_lane_7);
    results[3] = lower_half;
    results[4] = peers;
  }
}

int main() {
  uint32_t* results = nullptr;
  CUDA_CHECK(cudaMalloc(&results, 5 * sizeof(uint32_t)));

  warp_collectives_case<<<1, 32>>>(results);
  finish_kernel();

  uint32_t host[5]{};
  CUDA_CHECK(cudaMemcpy(host, results, sizeof(host), cudaMemcpyDeviceToHost));
  int status = require_equal("activemask", static_cast<int>(host[0]),
                             static_cast<int>(0xffffffffu));
  status |= require_equal("shuffle", host[1], 2);
  status |= require_equal("vote any", host[2], 1);
  status |= require_equal("ballot", host[3], 0x0000ffff);
  status |= require_equal("match", host[4], 0x11111111);

  CUDA_CHECK(cudaFree(results));
  return status;
}
