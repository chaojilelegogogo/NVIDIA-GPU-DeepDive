# 5.1 Part 0：Tensor Core 前的基础

## 5.1.1 GPU 怎样执行你的 kernel

CUDA 线程按连续 32 个组成一个 Warp。SM scheduler 每个周期从“就绪 Warp”中选择一个发射指令；Warp 内每条 lane 执行相同指令但处理不同数据。寄存器是每线程私有的最快存储，Shared Memory 是一个 CTA 共用的片上 SRAM，L2/HBM 保存全局数据。

```text
Global Memory / L2
        ↓  （普通 load、cp.async、TMA）
Shared Memory
        ↓  （ldmatrix、WGMMA descriptor、tcgen05 operand）
Registers / TMEM
        ↓
Tensor Core
        ↓
Registers / TMEM → Epilogue → Global Memory
```

Tensor Core 的意义并非让单线程更快，而是让一个协作线程集合以一条指令驱动专用矩阵乘加阵列。

## 5.1.2 从 GEMM 到 tile

GEMM 的数学形式：

```text
C[m, n] = Σk A[m, k] × B[k, n] + C[m, n]
```

朴素实现中每个线程只计算一个 `C[m,n]`，每次循环读一个 A 和一个 B。高性能实现将矩阵分块：

```text
整块 C
└─ CTA tile：一个 block 负责，例如 128×128
   └─ Warp/Warpgroup tile：一个协作集合负责，例如 64×64
      └─ Instruction tile：一条 MMA 负责，例如 m16n8k16
```

每次从 Global Memory 搬入 A/B tile 后，多个输出元素复用同一份输入。复用越高，越可能让 Tensor Core 持续工作；但 tile 更大也会消耗更多 Shared Memory、寄存器/TMEM 并要求更复杂的 pipeline。

## 5.1.3 三类 tile 不要混淆

| 名称 | 谁决定 | 例子 | 作用 |
|---|---|---|---|
| CTA tile | kernel 设计者/CUTLASS collective | 128×128×32 | 决定 block 工作量和 shared memory 用量 |
| Warp tile | Warp 或 Warpgroup 的分工 | 64×64×32 | 决定协作粒度、accumulator 压力 |
| Instruction tile | PTX MMA 指令 | `m16n8k16` | 硬件一次矩阵 FMA 的固定形状 |

`m16n8k16` 的含义是：指令将 A 的 16×16 子块和 B 的 16×8 子块相乘，累加到 16×8 的 C/D 子块。它不代表“每线程计算 16×8 个输出”；这些元素会分布在 Warp 的 lanes 和寄存器中。

## 5.1.4 初学者常见误解

- **“Warp 天然同步，所以不必同步。”** Volta 起 Independent Thread Scheduling 使此假设不安全。共享数据交接应使用正确的 `__syncwarp(mask)` 或 CTA barrier。
- **“Tensor Core 只要调用 WMMA 就会更快。”** 小矩阵、错误 layout、memory-bound kernel 或过多同步都可能更慢。
- **“Tensor Core 只做 FP16。”** 代际支持 FP16、INT8/INT4、TF32、BF16、FP8、FP4 等，但类型和指令形状严格受目标架构限制。
- **“一个 CUDA thread 发射的 Tensor 指令只影响自己。”** 多数 MMA 指令是 Warp/Warpgroup 集体操作；参与者必须满足指令的控制流和 operand 协议。

下一篇：[为什么需要 Tensor Core](./02-why-tensor-core.md)。
