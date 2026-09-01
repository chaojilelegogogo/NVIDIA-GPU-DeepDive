# 5.5 Part 4：Hopper —— TMA、`mbarrier` 与 `wgmma.mma_async`

## 5.5.1 Hopper 解决了哪三个问题

Ampere 很强，但仍有三个瓶颈：

1. `cp.async` 仍由许多线程各自算地址、发起小拷贝；
2. 单 Warp `mma.sync` 的 tile 和协作规模有限；
3. 更大 MMA 的 accumulator 持续占用寄存器。

Hopper 用 TMA、Warpgroup MMA 和异步完成协议分别处理前两个问题；第三个问题延续到 Blackwell TMEM。

## 5.5.2 TMA：由 DMA 搬一个 tile

TMA（Tensor Memory Accelerator）使用 Host 端构建的 tensor map 描述多维张量的 shape、stride、边界和 Shared Memory swizzle。一个线程可发起 tile 搬运，DMA 硬件完成地址生成和数据移动。

| 项目 | `cp.async` | TMA |
|---|---|---|
| 发起粒度 | 每线程小块 | 单线程描述一个多维 tile |
| 地址生成 | 软件逐线程计算 | 硬件读取 tensor map |
| PTX | `cp.async.*` | `cp.async.bulk.tensor.*` |
| completion | copy group | `mbarrier` transaction |

TMA 的“数据到了”不是普通 `__syncthreads()`：需要通过 `mbarrier` 的 phase/token 与 transaction completion 表达。TMA multicast 还可将 tile 发给同一 cluster 的多个 CTA；这要求正确的 cluster launch 与 DSM 生命周期。

## 5.5.3 Warpgroup：为什么从 32 线程变成 128 线程

Warpgroup 由 4 个连续 Warp（128 threads）组成。更大协作集合可驱动更大的 matrix instruction tile、分摊指令发射开销并提高输入复用。它不是一个新的通用 CUDA block，也不是“128 个线程天然有 barrier”；需要会合时仍使用对应的 Warp/CTA/cluster 原语。

## 5.5.4 `wgmma.mma_async`：异步矩阵乘加

| 字段 | 内容 |
|---|---|
| 计算 | `D=A×B+C`，通常使用更大的 M/N/K instruction shape |
| 协作对象 | Warpgroup；参与线程必须满足同一控制流和对齐要求 |
| operand | 依变体来自寄存器或 Shared Memory matrix descriptor |
| accumulator | 寄存器 |
| PTX | `wgmma.mma_async` |
| 典型 SASS | 常见 `HGMMA`/`QGMMA`，需实测 |

概念性 PTX：

```ptx
// 省略真实 shape、descriptor 和 operand tuple；仅展示协议顺序。
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned ...;
wgmma.commit_group.sync.aligned;
// 做独立工作或准备下一 stage
wgmma.wait_group.sync.aligned 0;
```

| 指令 | 含义 |
|---|---|
| `wgmma.fence` | 建立 WGMMA operand/accumulator 的专用依赖边界 |
| `wgmma.mma_async` | 发起异步 Tensor Core 计算，结果稍后才可读 |
| `commit_group` | 提交一批 WGMMA |
| `wait_group N` | 等待到最多 N 个较早 MMA group 未完成 |

`wait_group N` 不是“等第 N 组”；`wait_group 0` 只应放在首次读取对应 accumulator 或要覆盖其寄存器之前。WGMMA fence 也不能替代 generic↔async proxy fence。

## 5.5.5 Hopper GEMM pipeline

```text
TMA 把下一 A/B tile 写入 Shared Memory（含 swizzle）
  ↓
mbarrier 等待 transaction completion
  ↓
WGMMA 从 Shared descriptor 读取 operand
  ↓
commit group；计算下一 stage 或执行标量工作
  ↓
wait group；读取 register accumulator 做 epilogue
```

TMA completion 与 WGMMA completion 是独立依赖，前者等 `mbarrier`，后者等 `wgmma.wait_group`。完整同步协议见[第十三部分 13.3](../part13-synchronization-handbook/03-async-pipelines.md)和[13.4](../part13-synchronization-handbook/04-tensor-synchronization.md)。

下一篇：[Blackwell TMEM 和 `tcgen05`](./06-blackwell-tcgen05.md)。
