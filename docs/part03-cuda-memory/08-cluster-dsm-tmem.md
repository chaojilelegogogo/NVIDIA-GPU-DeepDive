# 3.10–3.11 Cluster/DSM 与 Blackwell TMEM：TMA 的上下游

> Cluster 的**执行层级**见[第二部分 2.2.4](../part02-cuda-programming-model.md)；`cg::this_cluster()` API 见[第七部分 7.4](../part07-cuda-cpp-api.md)；`cluster.sync()` 与可见性见[第十三部分 13.1.3](../part13-synchronization-handbook/01-execution-barriers.md)。**本节回答 Memory Hierarchy 问题：DSM 是什么、怎么寻址、性能如何。**

## 3.10.1 TMA 解决搬运，不解决所有存储问题

完整 Tensor pipeline 有三段：

```text
global tensor
  ── TMA ──→ shared/cluster shared operand
  ── Tensor Core ──→ accumulator
  ── epilogue/store ──→ global tensor
```

Hopper 的 Cluster/DSM 扩大 shared operand 的复用范围；Blackwell DC 的 TMEM 改变 accumulator 的存放位置。它们与 TMA 协同，但不是 TMA 本身。

## 3.10.2 Thread Block Cluster（记忆点，详解在 Part 02）

普通 CTA 可独立调度，无法假定同时驻留。Cluster 增加调度保证：一组 CTA 被共同调度到可低延迟通信的硬件范围，从而才能讨论 DSM 与 cluster barrier。

```cuda
__global__ void __cluster_dims__(2, 1, 1) kernel(...);
// 或 cudaLaunchAttributeClusterDimension
```

合法 cluster 大小见设备属性。本文件不再重复执行模型细节。

## 3.10.3 Distributed Shared Memory（DSM）

### 为什么需要 DSM？

传统模型下，每个 CTA 的 shared 对 peer CTA **不可见**：

```text
CTA0 shared ──X── CTA1
CTA1 shared ──X── CTA0
彼此只能经 Global / L2 通信
```

问题：

1. Block 之间无法直接读对方 shared 热数据；
2. Global memory 延迟远高于 shared，跨 CTA 复用成本高；
3. Tensor Core 计算需要更近的 operand——同一 A tile 若被多个 CTA 消费，反复经 Global 回灌浪费带宽。

Hopper DSM 不是“再造一块统一 Cluster SRAM”，而是：

```text
Cluster
├── CTA0 本地 shared（物理仍在 SM0）
│       ▲
│       │ cluster fabric / DSM 路径
│       ▼
└── CTA1 本地 shared（物理仍在 SM1）
```

每个 CTA 仍拥有并初始化自己的 shared；peer CTA 通过 **cluster 地址映射** 远程访问。

### DSM 地址模型

概念上，cluster 内每个 CTA 的 shared 有独立本地视图；`map_shared_rank` 把“本地 shared 指针 + peer rank”翻译成可访问 peer 的地址：

```text
CTA0 local shared 基址（示意）: 0x1000
CTA1 local shared 基址（示意）: 0x2000

CTA0 上：
  smem                          → 访问自己的 0x1000…
  map_shared_rank(smem, 1)      → 指向 CTA1 的 shared

CTA1 上：
  map_shared_rank(smem, 0)      → 指向 CTA0 的 shared
```

PTX/SASS 侧常见为 `.shared::cluster` 或目标相关的 remote shared 寻址；不要把 DSM 理解成“换了一个更大的 `__shared__` 关键字”。

### DSM 访问路径

```text
SM0 线程
  → 本地 Shared Memory Bank          （local shared）
  → Cluster Fabric / 跨 SM 互联
  → SM1 Shared Memory Bank           （remote DSM）
```

因此：

| 路径 | 典型延迟直觉 | 带宽直觉 |
|---|---|---|
| 本地 shared | 最低 | 最高（受 bank conflict 约束） |
| remote DSM | 介于 shared 与 L2/global 之间 | 次于本地 shared |
| global / L2 | 最高 | 受 HBM/L2 吞吐约束 |

经验规则：`latency(shared) < latency(DSM) < latency(global)`。DSM 适合**减少 global 往返或共享热点 tile**，不适合把本可本地完成的访问全部改成 remote。

### `map_shared_rank()` 与生命周期

```cuda
namespace cg = cooperative_groups;
cg::cluster_group cluster = cg::this_cluster();

extern __shared__ int smem[];
cluster.sync(); // 所有 CTA 已建立 shared，才能安全映射

const unsigned peer = /* 合法 rank */;
int* remote = cluster.map_shared_rank(smem, peer);
int x = remote[index];

cluster.sync(); // peer 完成访问后，owner 才能退出或复用该 storage
```

硬性规则：

1. **先 sync 再 remote 读**（或使用等价的明确发布协议）：保证 owner 已写好被读数据；
2. **peer 仍在访问时，owner 不得退出 / 复用**该 shared；
3. `map_shared_rank` 是地址翻译 API，**不是** barrier，也不发布任意 payload；
4. rank 必须落在 `[0, cluster.num_blocks())`，且对应 CTA 仍存活。

API 细节见[第七部分 `cluster_group`](../part07-cuda-cpp-api.md)；barrier 语义见[第十三部分](../part13-synchronization-handbook/01-execution-barriers.md)。

### DSM 性能模型（选型）

| 场景 | 倾向 |
|---|---|
| 同一 CTA 内 tile 复用 | 本地 shared + `__syncthreads` |
| 同 cluster 内少数 CTA 共享热点 | DSM 或 TMA multicast |
| 任意 CTA、无共同驻留保证 | Global + device-scope atomic/fence，或拆 kernel |
| 把所有跨 CTA 通信都改 DSM | 通常错误：remote 竞争与生命周期复杂度上升 |

## 3.10.4 TMA multicast 为什么依赖 Cluster

多个 CTA 需要同一个 A tile：

```text
无 multicast：
CTA0 TMA A → smem0
CTA1 TMA A → smem1
CTA2 TMA A → smem2

multicast：
一次 descriptor/tile 请求
  ├→ smem0
  ├→ smem1
  └→ smem2
```

目标仍是每个 CTA 的物理 shared buffer，因此需要 cluster topology、CTA mask、remote address/barrier 与生命周期规则。收益大小取决于原请求是否命中 L2、cluster 复用率和 multicast fan-out。

## 3.10.5 Blackwell CTA pair

数据中心 Blackwell 的 CTA pair 将两个 CTA/SM 组织成更紧密的 Tensor 协作单元。TMA 的 `.cta_group::2`、`tcgen05` CTA group 与 TMEM allocation 可以围绕 pair 建 pipeline。

必须区分：

- **TMA CTA group**：一次 tensor copy 的 destination/completion 归属；
- **Tensor CTA group**：`tcgen05` 操作的参与和资源范围；
- **Cluster**：更一般的 CTA 调度/DSM/同步范围。

三者可能重叠，但语义不能互换。

## 3.11.1 TMEM 为什么出现

Hopper WGMMA accumulator 位于 register：

```text
shared A/B → WGMMA → register accumulator
```

更大 tile 会显著增加 register pressure。Blackwell DC 引入 Tensor Memory：

```text
shared A/B → tcgen05 → TMEM accumulator
                           ↓
                     tcgen05.ld
                           ↓
                  register epilogue
```

TMEM 是 Tensor pipeline 的专用片上空间，不是可用普通指针读写的 shared memory。

## 3.11.2 TMEM 生命周期

```text
tcgen05.alloc
  → 获得 TMEM columns/address
  → TMA 准备 A/B/scale
  → required proxy fence
  → tcgen05.mma / tcgen05.cp
  → commit / wait
  → tcgen05.ld 做 epilogue
  → 所有消费者结束
  → tcgen05.dealloc
```

任何一步都不能被 `__syncthreads()` 简单替代：

- TMA mbarrier 回答 operand 是否到 shared；
- proxy fence 回答不同访问 proxy 的 ordering；
- `tcgen05` completion 回答 Tensor operation 是否产出 TMEM；
- CTA barrier 只回答线程会合及其定义的可见性。

## 3.11.3 `tcgen05.cp` 与 TMA 的区别

- TMA：主要连接 global 与 shared/cluster shared，TensorMap 负责多维地址和布局；
- `tcgen05.cp`：在 Tensor/TMEM 规定的片上路径中搬运矩阵或 scale-factor 数据；
- `tcgen05.mma`：消费 operand 并将 accumulator 写入 TMEM。

因此不能把“Blackwell 数据搬运”只画成 TMA：

```text
global --TMA--> shared --tcgen05.cp/mma--> TMEM --tcgen05.ld--> register
```

## 3.11.4 数据中心与消费级 Blackwell 分叉

| 目标 | TMA/TMEM 结论 |
|---|---|
| `sm_100a`/`sm_103a` 数据中心 Blackwell | 支持完整 `tcgen05`/TMEM，并有多项 Blackwell TMA 扩展 |
| `sm_120` RTX 50 系列 | 不支持 TMEM/`tcgen05`；不能因“Blackwell”品牌名推断数据中心 TMA/CTA-pair 能力 |

`sm_120` 应沿 `mma.sync` block-scale、普通 shared/register 与该目标实际支持的 copy 指令学习。编译时不要把 `-arch=sm_100a` 产物拿到 `sm_120` 运行。

## 3.11.5 一个跨代总表

| 代际 | Operand 搬运 | Tensor 计算 | Accumulator | 跨 CTA 复用 |
|---|---|---|---|---|
| Ampere | `cp.async` | `mma.sync` | register | 无 cluster |
| Hopper | TMA | WGMMA | register | Cluster/DSM/multicast |
| Blackwell DC | 扩展 TMA + CTA pair | `tcgen05` | TMEM | Cluster + CTA pair |
| Blackwell `sm_120` | 按该 target 可用 copy 路径 | block-scale `mma.sync` | register | 不按 DC 路径推断 |

`tcgen05` operand、TMEM 与 `sm_120` 分叉见[第五部分 Blackwell Tensor Core](../part05-tensor-core-handbook/06-blackwell-tcgen05.md)；Tensor completion 与 proxy ordering 见[第十三部分同步手册](../part13-synchronization-handbook/04-tensor-synchronization.md)。

下一篇：[正确性与性能诊断](./09-debug-performance.md)。
