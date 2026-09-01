#include "common.cuh"

/*
 * 目的：
 *   展示 atomic RMW 是否需要返回旧值，会影响编译器选择 atom 还是 red。
 *
 * 验证：
 *   两个 counter 初值均为 10；使用返回值的 atomicAdd 返回 10；
 *   两个 counter 最终都变为 11。
 *
 * 观察：
 *   old value 被使用 → 常见 atom.global.add
 *   old value 未使用 → 编译器可选择 red.global.add
 *   这是代码生成选择，不改变 atomic update 的正确性。
 */
__global__ void atom_vs_red_case(int* atom_counter,
                                 int* red_counter,
                                 int* old_value) {
  *old_value = atomicAdd(atom_counter, 1);
  atomicAdd(red_counter, 1);
}

int main() {
  int initial = 10;
  int* atom_counter = nullptr;
  int* red_counter = nullptr;
  int* old_value = nullptr;
  CUDA_CHECK(cudaMalloc(&atom_counter, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&red_counter, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&old_value, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(atom_counter, &initial, sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(red_counter, &initial, sizeof(int),
                        cudaMemcpyHostToDevice));

  atom_vs_red_case<<<1, 1>>>(atom_counter, red_counter, old_value);
  finish_kernel();

  int host_atom = 0;
  int host_red = 0;
  int host_old = 0;
  CUDA_CHECK(cudaMemcpy(&host_atom, atom_counter, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_red, red_counter, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_old, old_value, sizeof(int),
                        cudaMemcpyDeviceToHost));
  int status = require_equal("atomic old value", host_old, 10);
  status |= require_equal("atom counter", host_atom, 11);
  status |= require_equal("red counter", host_red, 11);

  CUDA_CHECK(cudaFree(old_value));
  CUDA_CHECK(cudaFree(red_counter));
  CUDA_CHECK(cudaFree(atom_counter));
  return status;
}
