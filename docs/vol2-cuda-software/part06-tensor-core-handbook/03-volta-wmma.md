# 6.3 Part 4：Volta —— WMMA 与第一代 `mma.sync`

> 先用 WMMA 学正确性与 Warp 协作，再学习 `mma.sync`。不要一开始手写 PTX。

## 6.3.1 Volta Tensor Core 能计算什么

Volta 首次提供 Tensor Core。CUDA 程序员看到的是 Warp 协作的矩阵 FMA：

```text
D = A × B + C
```

一条 PTX `mma.sync` 的 shape 例如 `m8n8k4`，描述一个 8×4 的 A 与 4×8 的 B 相乘，并累加到 8×8 输出。一个大 GEMM 会循环 K 维、组合多个 instruction tile，并把输出分别分给许多 Warp/CTA。

硬件资料中常以较小的矩阵阵列单元描述实现；这与 PTX `m8n8k4` 不是同一抽象层。编程时以 PTX ISA 的 shape/operand 表为准。

## 6.3.2 CUDA 层：`nvcuda::wmma`

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm(const half* A, const half* B, float* C,
                          int lda, int ldb, int ldc) {
  int warp = threadIdx.x / warpSize;
  // 省略 block/warp 对应的 A/B/C tile 指针计算
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

  wmma::fill_fragment(c, 0.0f);
  for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a, A + k, lda);
    wmma::load_matrix_sync(b, B + k * ldb, ldb);
    wmma::mma_sync(c, a, b, c);
  }
  wmma::store_matrix_sync(C, c, ldc, wmma::mem_row_major);
}
```

| API | 含义 | 初学者要检查的前提 |
|---|---|---|
| `fragment` | Warp 各 lane 上分布式的 A/B/accumulator 片段 | shape、数据类型、layout 必须匹配 |
| `fill_fragment` | 初始化 accumulator | 首次 MMA 前必须定义 C 值 |
| `load_matrix_sync` | cooperative load 到 fragment | 指针、leading dimension、对齐和 tile 边界正确 |
| `mma_sync` | `D=A×B+C` 累加 | Warp 所有参与 lane 必须一致调用 |
| `store_matrix_sync` | 将 accumulator fragment 写回 | 输出 layout 与 leading dimension 正确 |

`fragment` 不是“每个线程拥有完整 16×16 矩阵”。其元素如何分布在 32 lanes 是实现细节；正确做法是只使用 WMMA API，不手工假设 fragment 内数组的语义。

## 6.3.3 PTX 层：`wmma.*` 与 `mma.sync`

WMMA C++ 可生成 `wmma.load`/`wmma.mma`/`wmma.store` PTX，也可能被编译器降为相关 `mma.sync` 形式。不要把具体生成方式当作 API 契约。

较低层的概念形式：

```ptx
// 仅说明数据流；不能直接复制，真实 operand tuple 以 PTX ISA 为准。
wmma.load.a.sync.aligned.row.m16n16k16.global.f16  ...;
wmma.load.b.sync.aligned.col.m16n16k16.global.f16  ...;
wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16.f32 ...;
wmma.store.d.sync.aligned.row.m16n16k16.global.f32 ...;
```

Volta 还公开了更基础的 `mma.sync.aligned.m8n8k4...`。它接受每 lane 的寄存器 tuple；Warp 可被理解为 Quad Pair 等内部协作分组。手写此类 PTX 时，A/B/C/D tuple 的数量、类型、layout 和所有参与 lane 的执行路径必须精确符合 PTX ISA。

## 6.3.4 SASS 和硬件验证

Volta/Turing/Ampere 的 Tensor Core SASS 常出现 `HMMA` 线索；但命名、shape 编码与调度细节依 Toolkit/目标变化。建议：

```bash
nvcc -arch=sm_70 -ptx wmma_gemm.cu -o wmma-sm70.ptx
nvcc -arch=sm_70 -cubin wmma_gemm.cu -o wmma-sm70.cubin
nvdisasm -c wmma-sm70.cubin
```

先验证数值结果，再观察 PTX/SASS。若未见预期指令，依次检查：目标架构、数据类型、tile shape/对齐、WMMA API 前提与编译器版本。

下一篇：[Turing/Ampere 的 `mma.sync`、`ldmatrix` 与 `cp.async`](04-turing-ampere-mma.md)。
