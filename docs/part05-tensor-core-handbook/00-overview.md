# 第五部分 Tensor Core 指令全景与编程手册

> 面向 GPGPU 初学者。本部分按“为什么这代必须改变上一代的做法”组织，而不是按 API 名字堆砌。最终目标是能读懂一个 GEMM 的 CUDA/CuTe 代码，知道它会使用哪类 PTX 指令、数据放在哪里、谁在等待谁，以及如何用 PTX/SASS 验证自己的判断。

## 本部分的学习路线

```text
Part 0  基础：SIMT、warp、GEMM tile、寄存器与 shared memory
  ↓
Part 1  为什么需要 Tensor Core：标量 FMA 的吞吐与 Roofline 瓶颈
  ↓
Part 2  Volta：WMMA 和第一代 mma.sync
  ↓
Part 3  Turing/Ampere：mma.sync、ldmatrix、cp.async
  ↓
Part 4  Hopper：TMA、mbarrier、WGMMA、warpgroup
  ↓
Part 5  Blackwell：TMEM、tcgen05、block scaling、CTA group
  ↓
Part 6  CUDA → PTX → SASS：如何实证，而非猜测
  ↓
Part 7  CUTLASS/CuTe 和七阶 GEMM 实战路线
```

## 三个必须先接受的事实

1. Tensor Core 执行的是**固定形状的矩阵 multiply-accumulate**，不是任意大小的 `matmul` API。大矩阵由许多 instruction tile 组合而成。
2. 一个 Tensor 指令由 Warp、Warpgroup 或目标指令规定的协作集合共同完成；单个线程只持有一小部分 operand/结果。
3. CUDA API、PTX、SASS 不是稳定的一一映射。本文中的 SASS 名称是定位线索；必须在指定 CUDA Toolkit、`-arch` 与 GPU 上实际反汇编。

## 架构主线速览

| 代际 | CUDA 编程入口 | PTX 核心指令 | 数据路径 | 新解决的问题 |
|---|---|---|---|---|
| Volta | `nvcuda::wmma` | `wmma.*` / `mma.sync` | register → Tensor Core → register | 用矩阵阵列替代标量 CUDA Core FMA |
| Turing/Ampere | WMMA、inline PTX、CUTLASS atom | `mma.sync`、`ldmatrix` | shared → register → Tensor Core | 可控 operand layout、更多类型/shape |
| Ampere | `cuda::memcpy_async` | `cp.async` | global → shared | 去掉寄存器中转并搬运/计算重叠 |
| Hopper | CUTLASS/CuTe | TMA、`mbarrier`、`wgmma.mma_async` | global → shared descriptor → Tensor Core → register | 更大 tile、异步 Tensor pipeline |
| Blackwell DC | CuTe/CUTLASS / inline PTX | `tcgen05.*` | shared → Tensor Core → TMEM | accumulator 不再挤压 register file |
| Blackwell `sm_120` | CUTLASS/CuTe | block-scale `mma.sync` 变体 | register-oriented | 消费级低精度 MMA，不含 TMEM/tcgen05 |

## 文件导航

| 文件 | 内容 |
|---|---|
| [01-foundations.md](./01-foundations.md) | SIMT、GEMM、tile、数据路径与术语 |
| [02-why-tensor-core.md](./02-why-tensor-core.md) | 标量 FMA、Roofline、mixed precision |
| [03-volta-wmma.md](./03-volta-wmma.md) | WMMA API、fragment、Volta `mma.sync` |
| [04-turing-ampere-mma.md](./04-turing-ampere-mma.md) | `mma.sync`、`ldmatrix`、`cp.async` |
| [05-hopper-wgmma.md](./05-hopper-wgmma.md) | TMA、`mbarrier`、WGMMA、warp specialization |
| [06-blackwell-tcgen05.md](./06-blackwell-tcgen05.md) | TMEM、`tcgen05`、`sm_120` 分叉 |
| [07-mapping-cutlass-gemm.md](./07-mapping-cutlass-gemm.md) | CUDA/PTX/SASS、CUTLASS/CuTe、GEMM 实战 |

同步协议不在本章重复展开：`cp.async`、TMA、WGMMA、`tcgen05` 的 wait/fence/transaction 语义请配合[第十三部分同步手册](../part13-synchronization-handbook/00-overview.md)阅读。
