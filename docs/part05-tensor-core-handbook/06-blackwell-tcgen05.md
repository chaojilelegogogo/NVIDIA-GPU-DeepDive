# 5.6 Part 5：Blackwell —— TMEM、`tcgen05` 与 block scaling

> “Blackwell”不是一套单一的可编程 Tensor ISA。本章先分数据中心与消费级路径，再讨论指令。

## 5.6.1 必须先区分目标

| 目标 | 核心 Tensor 路径 | 能否使用 TMEM/`tcgen05` |
|---|---|---|
| 数据中心 Blackwell，例如 `sm_100a`/`sm_103a` | TMA + TMEM + `tcgen05` | 可以，依 Toolkit/PTX ISA |
| RTX 50 等消费级 `sm_120` | Warp `mma.sync` 的 block-scale 扩展 | 不可以 |

将 `tcgen05.*` 编译到 `sm_120` 是错误方向；首先让 `-arch` 和 PTX ISA 决定可用路径。

## 5.6.2 为什么需要 TMEM

Hopper WGMMA 的 accumulator 位于寄存器。tile 增大后，accumulator 消耗更多寄存器，降低 occupancy 或限制可选 tile。数据中心 Blackwell 引入 Tensor Memory（TMEM），让 Tensor Core 累加结果驻留在专用片上空间：

```text
Hopper：shared operand → WGMMA → register accumulator
Blackwell DC：shared operand → tcgen05 → TMEM accumulator → ld → register epilogue
```

这并不等价于“TMEM 是普通 Shared Memory”：它有显式分配/释放、特定访问指令和严格的 Tensor pipeline 同步要求。

## 5.6.3 `tcgen05` 指令族如何使用

按生命周期理解，不要孤立背 mnemonic：

| 阶段 | 指令族 | 含义 | 什么时候用 |
|---|---|---|---|
| 分配 | `tcgen05.alloc` | 获取 TMEM 空间 | 使用 Tensor accumulator 前 |
| 计算 | `tcgen05.mma` | 发起矩阵 FMA，结果累加到 TMEM | A/B tile 已可供 Tensor 路径读取后 |
| 数据移动 | `tcgen05.cp` | 在规定路径间搬运 Tensor 相关数据 | 准备/转换 Tensor 数据时 |
| 读写 | `tcgen05.ld` / `st` | TMEM 与寄存器/规定路径交换 | epilogue 或后续计算需要结果时 |
| 完成 | `commit` / `wait` / `fence` | 提交异步工作、等待结果、建立 required ordering | 结果被读取、复用或覆盖之前 |
| 释放 | `tcgen05.dealloc` | 归还 TMEM | 所有消费者结束后 |

概念 stage：

```text
alloc TMEM
  → TMA/Shared Memory 准备 A、B，并等待 transaction completion
  → 必要的 async-proxy fence
  → tcgen05.mma / tcgen05.cp
  → commit
  → wait（首次读 TMEM、复用/释放资源前）
  → tcgen05.ld 到寄存器做 epilogue
  → dealloc
```

真实 `tcgen05` operand list、CTA group 限制、shape、scale 以及 fence/wait 形式会随 PTX ISA 版本变化。上图是协议框架，不是可复制的 PTX 程序；手写前必须查当前 Toolkit 的 PTX ISA。

## 5.6.4 Single Thread Issue、CTA group、register group

某些 `tcgen05.mma` 变体可由单个线程发起。原因是输出位于 TMEM，而不再固定属于发起 Warpgroup 的寄存器 tuple。这只意味着**发起责任**可解耦；并不表示：

- 一个线程可独自完成整个 Tensor Core 操作；
- 不需要 TMEM allocation 或 TMA completion；
- 不需要 CTA group/topology 约束；
- `commit/wait` 可用 CTA barrier 替代；
- 任何寄存器都自动在 CTA 内共享。

CTA group 和 register group 是目标指令的资源/协作约束，不能自行推广为 CUDA 通用同步原语。详见第十三部分 13.4。

## 5.6.5 FP4、MXFP 与 block scaling

FP4/FP8 等低精度输入必须配合 scale 才能覆盖实际张量的局部数值范围。程序层面要将 scale 当作 operand 的一部分：

```text
高精度值 → 分块量化值 + 每块 scale
         → MMA（按指令规定读取 data/scale）
         → 高精度 accumulator / epilogue
```

不要假设“FP4”在所有库/硬件上的格式、block 大小或精度相同；明确格式、scale 类型、layout、accumulator 和误差标准。

## 5.6.6 消费级 `sm_120` 的正确学习路径

`sm_120` 没有 TMEM/`tcgen05`，应学习 `mma.sync.aligned.block_scale` 等可用 warp MMA 变体：

1. 先完成 FP16/BF16 `mma.sync` + `ldmatrix` GEMM；
2. 使用支持 `sm_120` 的 CUTLASS/CuTe 示例；
3. 导出 PTX，确认实际生成 block-scale MMA，而不是假定 `tcgen05`；
4. 对比 FP16/FP8/FP4 的误差、吞吐、寄存器与带宽；
5. 将 scale layout 纳入 Shared Memory swizzle 与 pipeline 设计。

下一篇：[CUDA→PTX→SASS、CUTLASS 与 GEMM 实战](./07-mapping-cutlass-gemm.md)。
