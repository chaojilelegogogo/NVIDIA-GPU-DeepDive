#include "common.cuh"

/*
 * Case 01：最小 FFMA 探针（CUDA → PTX → SASS）
 *
 * CUDA：
 *   d[i] = __fmaf_rn(a[i], b[i], c[i]);   // fused multiply-add
 *
 * 预期 PTX（Blackwell / sm_110 一类目标上常见）：
 *   ld.global.f32 ...
 *   fma.rn.f32 %fD, %fA, %fB, %fC;
 *   st.global.f32 ...
 *
 * 预期 SASS（以本机 cuobjdump 为准）：
 *   LDG ...
 *   FFMA R?, R?, R?, R?
 *   STG ...
 *
 * 运行时只验证数值正确性：1.5 * 2.0 + 3.0 == 6.0
 */
__global__ void ffma_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            const float* __restrict__ c,
                            float* __restrict__ d) {
  const int i = static_cast<int>(threadIdx.x);
  d[i] = __fmaf_rn(a[i], b[i], c[i]);
}

int main() {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::cout << "GPU: " << properties.name << "  sm_" << properties.major
            << properties.minor << '\n';

  constexpr int kN = 32;
  float host_a[kN], host_b[kN], host_c[kN], host_d[kN]{};
  for (int i = 0; i < kN; ++i) {
    host_a[i] = 1.5f;
    host_b[i] = 2.0f;
    host_c[i] = 3.0f;
  }

  float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr,
        *device_d = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, sizeof(host_a)));
  CUDA_CHECK(cudaMalloc(&device_b, sizeof(host_b)));
  CUDA_CHECK(cudaMalloc(&device_c, sizeof(host_c)));
  CUDA_CHECK(cudaMalloc(&device_d, sizeof(host_d)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_c, host_c, sizeof(host_c), cudaMemcpyHostToDevice));

  ffma_kernel<<<1, kN>>>(device_a, device_b, device_c, device_d);
  finish_kernel();

  CUDA_CHECK(cudaMemcpy(host_d, device_d, sizeof(host_d), cudaMemcpyDeviceToHost));

  int status = EXIT_SUCCESS;
  constexpr float kExpected = 6.0f;
  for (int i = 0; i < kN; ++i) {
    if (std::fabs(host_d[i] - kExpected) > 1e-6f) {
      std::cerr << "lane " << i << ": got " << host_d[i] << ", expected "
                << kExpected << '\n';
      status = EXIT_FAILURE;
      break;
    }
  }
  if (status == EXIT_SUCCESS) {
    std::cout << "ffma: all lanes saw 1.5*2.0+3.0 = 6.0\n";
  }

  CUDA_CHECK(cudaFree(device_d));
  CUDA_CHECK(cudaFree(device_c));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_a));
  return status;
}
