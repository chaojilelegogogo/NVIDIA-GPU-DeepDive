# 3.6–3.7 数据搬运演进：为什么从 `ld/st` 走到 TMA

## 3.6 Tiled GEMM 给出的搬运动机

朴素 GEMM 反复从 global 读取相同 A/B 元素；tiled GEMM 先将 tile 搬进 shared，使一次片外读取被 CTA 内多次 MMA/FMA 复用。优化后新的瓶颈变成：**谁负责把 tile 搬入 shared，以及搬运能否与计算重叠。**

```text
naive global load
  → shared-memory tiling
  → 去掉寄存器中转
  → 去掉逐线程地址生成
  → 跨 CTA 复用
  → accumulator 脱离 register
```

## 3.7 数据搬运基线：`ld.global` + `st.shared`

```cuda
float x = gmem[index]; // global → register
smem[local] = x;       // register → shared
__syncthreads();
```

数据路径：

```text
HBM/L2/L1 → LDG → thread register → STS → shared
```

代价不只是两条指令：

- 临时值占寄存器；
- 每个线程计算地址；
- warp scheduler 发射 load 和 store；
- load 返回后才能执行依赖的 store；
- 软件还要显式组织双缓冲和边界分支。

这条路径仍然重要：小规模、不规则、需要寄存器变换的数据不一定适合异步 copy。

## 3.E.1 Ampere：非 bulk `cp.async`

```text
global → shared
```

跳过通用寄存器中转，并允许 copy 与计算重叠。但工作分配仍是逐线程的：每个 lane 计算自己的 global/shared 地址并发起 4/8/16B copy。它解决“寄存器中转 + 同步等待”，没有彻底解决“每线程地址生成和大量小指令”。

## 3.E.2 Hopper：bulk copy

`cp.async.bulk` 把一次操作扩大到一段以字节数描述的连续区域，通常由少量线程发起。相比非 bulk `cp.async`：

- size 是运行时 32-bit 字节数，但须满足 16B 粒度等约束；
- load 方向通常以 mbarrier transaction 完成；
- store 方向通常以 bulk async-group 完成；
- 支持 global/shared/cluster 间若干方向，具体以语法变体为准。

它适合连续块，却仍没有表达多维 tensor 的 shape/stride/boundary。

## 3.E.3 Hopper：TMA / tensor copy

TMA 把不随 tile 坐标变化的元数据放进 128B `CUtensorMap`：

```text
global base
global dimensions and byte strides
box dimensions
element traversal strides
element type
interleave / swizzle / L2 promotion / OOB fill
```

kernel 每次只给 descriptor、tile 坐标、shared 目标和 completion object。硬件展开 1D–5D 地址、处理边界并按 swizzle 落入 shared。

这一步的本质是：

```text
逐 lane 描述“搬哪些字节”
        ↓
单个 issuer 描述“我要哪个 tensor tile”
```

## 3.E.4 Blackwell：扩展“搬什么”和“如何布局”

Blackwell 不是把基础 TMA 推倒重来，而是在 TensorMap/tensor-copy 模型上扩展：

- U4/U6 等 sub-byte tensor element 与 packing/padding；
- 128B swizzle 的 32B/64B atomicity 和 8B flip 等模式；
- `tile::gather4` / `tile::scatter4`；
- `im2col::w` / `im2col::w::128`；
- CTA pair 的 `.cta_group::2` 与 peer mbarrier 交接；
- 更多 load/store 方向和目标差异；
- bulk/tensor reduce、L2 prefetch、cache hint；
- 新 PTX 中的 `.ignore_oob`、`.sem/.scope` 等属于 bulk 指令族持续演进，需区分 PTX 版本和硬件 target。

这些能力减少的不只是 copy 指令，还包括 gather/scatter、卷积展开、低精度 unpack/layout、跨 CTA 通知等数据编排工作。

## 3.E.5 选择决策树

```text
要搬的数据是否 global → shared？
├── 否 → 普通 ld/st、bulk store/reduce 或其它专用路径
└── 是
    ├── 很小、不规则、搬运时要做任意变换 → 普通 ld/st
    ├── 每线程自然对应 4/8/16B，目标 Ampere+ → cp.async
    ├── 一大段连续字节，目标 Hopper+ → cp.async.bulk
    └── 规则的 1D–5D tile / 边界 / swizzle / im2col → TMA
```

再问四个性能问题：

1. tile 是否足够大，能摊薄 descriptor 和发起开销？
2. 是否有足够计算覆盖 copy latency？
3. shared 容量能否容纳 2–4 个 stage？
4. 单线程 producer 会不会因调度不及时成为 pipeline 气泡？

TMA 不是自动更快。小 tile、低复用、同步位置错误或 producer warp 调度不足，都可能让它输给简单 load。

## 3.E.6 对照表

| 机制 | 最低代际 | 发起粒度 | 地址生成 | 寄存器中转 | 典型完成 |
|---|---|---|---|---|---|
| `ld.global` + `st.shared` | 通用 | 每线程元素/向量 | 每线程 | 有 | 普通依赖 + CTA barrier |
| `cp.async` | Ampere | 每线程 4/8/16B | 每线程 | 无 | non-bulk async group |
| `cp.async.bulk` | Hopper | 连续大块 | base + size | 无 | mbarrier 或 bulk group |
| `cp.async.bulk.tensor` / TMA | Hopper | 1D–5D tile | TensorMap 硬件展开 | 无 | load:mbarrier；store:bulk group |
| Blackwell tensor modes | Blackwell 特定 target | tile/gather/im2col/sub-byte | descriptor + mode | 无 | mbarrier/bulk group/CTA group |

## 3.E.7 GEMM 中的完整因果链

```text
朴素 GEMM：每次 FMA 都读 global
  → tiled GEMM：shared 复用，减少 HBM 流量
  → cp.async：搬运不再经过 register，并与 MMA 重叠
  → TMA：单 issuer + descriptor，减少地址/指令开销并自动 swizzle
  → multicast：cluster CTA 复用相同 global tile
  → Blackwell TMEM：accumulator 脱离 register file
```

下一篇：[Ampere `cp.async` 使用手册](./04-ampere-cp-async.md)。
