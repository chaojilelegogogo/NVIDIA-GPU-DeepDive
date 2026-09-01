#include "common.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

#include <utility>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult result__ = (call);                                              \
    if (result__ != CUDA_SUCCESS) {                                          \
      const char* name__ = nullptr;                                          \
      cuGetErrorName(result__, &name__);                                     \
      std::cerr << "Driver API error: " << (name__ ? name__ : "unknown")    \
                << '\n';                                                     \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (false)

/*
 * 目的：
 *   展示 Hopper+ TMA load 如何把“线程 arrival”和“异步 transaction bytes”
 *   同时记入一个 mbarrier phase。
 *
 * 验证：
 *   一个线程发起 1D TMA，将 128 个 int 从 global 搬到 shared；所有线程等待
 *   barrier token 后复制到 output，结果应与 input 完全一致。
 *
 * 观察：
 *   cp.async.bulk.tensor.1d...mbarrier::complete_tx::bytes
 *   mbarrier arrive/expect_tx/try_wait，以及目标相关 TMA SASS。
 */
__global__ void tma_transaction_case(
    const __grid_constant__ CUtensorMap tensor_map, int* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int count = 128;
  __shared__ alignas(128) int tile[count];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier_t barrier;

  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  barrier_t::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_1d_global_to_shared(
        tile, &tensor_map, 0, barrier);
    token = cuda::device::barrier_arrive_tx(
        barrier, 1, sizeof(tile));
  } else {
    token = barrier.arrive();
  }
  barrier.wait(std::move(token));

  output[threadIdx.x] = tile[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    (&barrier)->~barrier_t();
  }
#else
  if (threadIdx.x == 0) {
    output[0] = -1;
  }
#endif
}

int main() {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 9) {
    std::cout << "TMA requires compute capability 9.0+: skipped\n";
    return EXIT_SUCCESS;
  }

  constexpr int count = 128;
  int host_input[count];
  for (int i = 0; i < count; ++i) {
    host_input[i] = 1000 + i;
  }

  int* input = nullptr;
  int* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, sizeof(host_input)));
  CUDA_CHECK(cudaMalloc(&output, sizeof(host_input)));
  CUDA_CHECK(cudaMemcpy(input, host_input, sizeof(host_input),
                        cudaMemcpyHostToDevice));

  CU_CHECK(cuInit(0));
  CUtensorMap tensor_map{};
  uint64_t global_dim[1] = {count};
  // CUDA 13 / Thor 上 1D encode 不能传 nullptr stride；给元素字节步长。
  uint64_t global_stride[1] = {sizeof(int)};
  uint32_t box_dim[1] = {count};
  uint32_t element_stride[1] = {1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_INT32,
      1,
      input,
      global_dim,
      global_stride,
      box_dim,
      element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  tma_transaction_case<<<1, count>>>(tensor_map, output);
  finish_kernel();

  int host_output[count]{};
  CUDA_CHECK(cudaMemcpy(host_output, output, sizeof(host_output),
                        cudaMemcpyDeviceToHost));
  int status = EXIT_SUCCESS;
  for (int i = 0; i < count; ++i) {
    if (host_output[i] != host_input[i]) {
      std::cerr << "TMA mismatch at " << i << '\n';
      status = EXIT_FAILURE;
      break;
    }
  }
  std::cout << "TMA transaction: "
            << (status == EXIT_SUCCESS ? "passed" : "failed") << '\n';

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  return status;
}
