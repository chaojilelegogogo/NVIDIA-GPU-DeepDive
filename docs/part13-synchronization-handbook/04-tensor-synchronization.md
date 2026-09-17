# 13.4 Tensor Core Synchronization：数据依赖不是 CTA barrier

> `mma.sync` 的 sync、WGMMA 的 async 和 `tcgen05` 的 commit/wait 处在不同代际的 Tensor 数据依赖模型中。它们描述“结果何时可安全消费”，不自动解决其它 CTA 或其它 proxy 的一般通信。

## 13.4.1 `mma.sync`：warp 集体、同步完成

| 字段 | 内容 |
|---|---|
| 架构 | Volta+，形状和数据类型随代际扩展 |
| 同步对象 | 一个 warp 的参与 lanes |
| Execution barrier | 指令是同步式：结果按指令数据依赖进入后续寄存器使用 |
| Memory ordering | 不提供 general-purpose CTA/global fence |
| CUDA | WMMA `mma_sync`，或 C++/内联 PTX 路径 |
| PTX | `mma.sync...` |
| 典型 SASS | `HMMA` / 目标相关 MMA 助记符 |

`mma.sync` 的 `.sync` 表示其 warp 协作执行要求与同步结果语义，不能解释为“替所有线程做 `__syncthreads()`”。若操作数由 shared memory 填充，仍须在装载者与 `ldmatrix`/读取者之间建立适当的 warp 或 CTA 同步。

## 13.4.2 Hopper `wgmma.mma_async`

| 字段 | 内容 |
|---|---|
| 架构 | Hopper `sm_90`/`sm_90a` |
| 协作对象 | 一个 warpgroup（4 个连续 warp，128 threads）以一致控制流发射 |
| 操作数 | register 或 shared-memory matrix descriptor；常由 TMA 准备 |
| 结果 | accumulator 寄存器，异步 Tensor Core 管线完成 |
| PTX 协议 | `wgmma.fence` → `wgmma.mma_async`* → `wgmma.commit_group` → `wgmma.wait_group N` |
| 典型 SASS | 常见 `HGMMA`/`QGMMA`，需反汇编确认 |

```ptx
// 示意，不是可直接粘贴的完整 operand 列表
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16...;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
// 此后才能读取相应 accumulator
```

**`fence` 的角色**：它是 WGMMA operand/accumulator 的专用依赖协议部分，确保此前寄存器/共享操作数修改与 WGMMA 之间符合要求；不要以它替换 generic/async proxy 之间需要的 `fence.proxy`。WGMMA 在 PTX memory model 中使用 async proxy，但这不意味着存在一个可泛化到所有 Tensor 指令的“tensor proxy”。

**`wait_group N` 的角色**：限制仍可未完成的 WGMMA group 数；只有在要读取对应 accumulator 或覆盖其寄存器前才等待到足够严格的级别。过早 `wait_group 0` 会消除计算重叠，过晚读取 accumulator 则为数据竞争。

WGMMA 不是一个“warpgroup barrier”。它不要求或保证 CTA 内其它 warp 到达某个位置。TMA → WGMMA 的正确配合还必须先以 mbarrier transaction 确认 shared tile 对 async/Tensor 路径可用，并在 proxy 转换处遵守 fence 规则。

## 13.4.3 Blackwell `tcgen05`：TMEM 与 Tensor completion

> 本节只适用于数据中心 Blackwell 的架构特性目标（如 `sm_100a`/`sm_103a`），不适用于消费级 `sm_120`。

| 字段 | 内容 |
|---|---|
| 目的 | 将 Tensor Core 累加结果置于 TMEM，解耦发起线程和寄存器 accumulator 所有者 |
| 主要 PTX | `tcgen05.alloc/dealloc`、`mma`、`cp`、`ld/st`、`commit`、`wait`、相关 fence |
| 同步对象 | Tensor 操作组及其 completion；具体参与者约束由所选 `tcgen05` 变体定义 |
| Execution barrier | `wait` 等待相关 Tensor 工作，不是 CTA/cluster 全员会合 |
| Memory ordering | TMEM/shared/async 路径的交接依赖该版本 PTX 规定的 fence/commit/wait 序列；不要自行命名统一 tensor proxy |
| CUDA | 无稳定、一对一公开 CUDA C++ API；通常通过 CuTe/CUTLASS 或内联 PTX |
| SASS | `TCGEN05.*` 或目标相关编码，必须用相应 Toolkit 的 `nvdisasm` 验证 |

推荐把一个 stage 写成显式协议，而非散落的指令：

```text
分配 TMEM
  → 确认 TMA/shared 操作数完成并完成必要 proxy fence
  → 发起 tcgen05 MMA/CP
  → commit Tensor 工作
  → 在读 TMEM、重用 TMEM 或依赖结果前 wait
  → tcgen05.ld 取回结果 / 进入下一 stage
  → 完成所有使用后 dealloc
```

`commit` 不是完成，`wait` 也不是 cluster barrier。CTA Pair/cluster 共享路径还必须符合 launch topology、TMA/DSM 与 cluster 同步要求。跨路径时优先遵照对应 CUDA Toolkit 的 `tcgen05` PTX ISA “required synchronization”段落；该指令族仍在快速演进，不能从早期示例复制到不同 Toolkit。

## 13.4.4 Tensor 完成的选型

| 计算路径 | 等待什么 | 不该用什么替代 |
|---|---|---|
| `mma.sync` | 普通寄存器数据依赖 | 无意义的 CTA barrier |
| `wgmma.mma_async` | `wgmma.wait_group`，在 consumer 点等待 | 仅 `__syncthreads()` |
| TMA 供给 WGMMA | `mbarrier` transaction + 必要 proxy fence | 仅 WGMMA wait |
| `tcgen05` | 该指令族要求的 commit/fence/wait | 仅 mbarrier 或普通 fence |

## 13.4.5 配套案例与目标限制

- [`case01_mma_sync.cu`](../../src/part13-synchronization-handbook/04-tensor-synchronization/case01_mma_sync.cu) 提供可运行的 WMMA/`mma.sync` 基线，验证 warp 集体计算与普通寄存器结果依赖。
- WGMMA 只能在支持该 ISA 的 Hopper 目标上以合法的 128-thread operand/descriptor 布局测试。
- `tcgen05`/TMEM 只在相应数据中心 Blackwell architecture-specific target 上测试；`sm_120` 不支持。

本仓库不会放置“只有 mnemonic、operand/layout 不合法”的假 WGMMA/tcgen05 kernel。目标相关实验说明见 [`src/.../04-tensor-synchronization/README.md`](../../src/part13-synchronization-handbook/04-tensor-synchronization/README.md)，完整 operand 构造应复用 Part 5 的 CuTe/CUTLASS 路线。

下一篇：[Collectives and Atomics](./05-collectives-and-atomics.md)。
