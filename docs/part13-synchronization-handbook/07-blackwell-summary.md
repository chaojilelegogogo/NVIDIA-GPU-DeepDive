# 13.7 Blackwell Synchronization Summary

## 13.7.1 先分清两条 Blackwell 路径

| 能力 | 数据中心 Blackwell `sm_100a`/`sm_103a` | 消费级 Blackwell `sm_120` |
|---|---|---|
| `tcgen05` / TMEM | 支持 | 不支持 |
| Tensor completion 协议 | `tcgen05` 的 fence/commit/wait | `mma.sync` 路径的普通寄存器依赖 |
| WGMMA/Hopper 风格概念 | 需要按目标 Toolkit/架构能力核对 | 不要假定等同数据中心 Blackwell |
| `cp.async` 基线 | 可用，取决于 ISA/代码生成条件 | 可用，取决于 ISA/代码生成条件 |
| cluster/TMA | 按硬件和 Toolkit 能力核对 | 不应从“Blackwell”品牌名直接推断可用 |

因此“Blackwell 推荐什么同步”没有单一答案；先写下 `-arch` 和实际 GPU capability，再选择协议。

### 13.7.1.1 消费级 `sm_120`：默认可依赖的同步子集

在尚未用 `cudaDeviceGetAttribute` / PTX ISA 核实之前，消费级 Blackwell 应默认只依赖下表“是”列；“否/勿假定”列不要从品牌名推断可用。

| 同步类别 | `sm_120` 默认可用？ | 说明 |
|---|---|---|
| Warp / CTA barrier（`__syncwarp` / `__syncthreads` / `bar.*`） | 是 | 全架构基线 |
| Order + Scope（cta/gpu/sys）+ `__threadfence*` | 是 | cluster scope 仍需能力探测 |
| Warp collective（`shfl/vote/match.sync`） | 是 | 使用 `.sync` 变体 |
| Atomic `atom` / `red` | 是 | order/scope 按参与者选择 |
| `cp.async` group / `cuda::pipeline` | 通常是 | 以代码生成条件为准；用 PTX/SASS 核实 |
| Cluster / DSM / TMA / WGMMA | 勿假定 | 必须按设备能力与 Toolkit 跳过或分支 |
| `tcgen05` / TMEM | 否 | 仅数据中心 `sm_100a`/`sm_103a` 一类目标 |

配套 case 对 cluster/TMA 等会在运行时打印 `skipped`；构建 `tcgen05` 探针时应显式 `-arch=sm_100a`（或等价），不要用 `sm_120` 冒充。

## 13.7.2 新增、替代与仍保留

| 类别 | 结论 | 工程建议 |
|---|---|---|
| CTA / warp barrier | 仍是基础 | 用最小正确参与范围；Volta+ 显式 `__syncwarp(mask)` |
| `membar` | 旧 PTX 表达仍常见 | 阅读旧代码时理解；新 PTX 按版本采用 `fence` 内存模型 |
| `fence.proxy.async` | generic↔async proxy 交接的重要组成 | 仅在发生 proxy 切换且规范要求时使用；不替代 completion；不存在统一 tensor proxy |
| `cp.async` | 仍是小粒度异步 copy 的重要基线 | 用 group pipeline；避免 commit 后立即 `wait_group 0` |
| TMA + `mbarrier` | Hopper+ 的 tile DMA/完成通知核心 | 将 phase、transaction bytes 和 cluster 生命周期显式设计 |
| WGMMA | Hopper 的 async Tensor dependency 模型 | 用 WGMMA 专用 fence/commit/wait，不以 CTA barrier 代替 |
| `tcgen05` | 数据中心 Blackwell 的新 Tensor/TMEM 模型 | 严格按当前 PTX ISA 的 required synchronization；不要移植至 `sm_120` |
| `vote`（无 `.sync`） | 已弃用 | 使用 `vote.sync` 和 CUDA `*_sync` intrinsics |

## 13.7.3 从需求到原语的决策树

```text
需要的“完成”到底是什么？
├── 多个线程必须走到同一位置
│   ├── 同 warp → __syncwarp(mask)
│   ├── 同 CTA → __syncthreads() / cuda::barrier
│   ├── 同 cluster → cluster.sync()（满足 launch 前提）
│   └── 全 grid → cooperative grid.sync() 或拆 kernel
├── 某个写要被另一观察者有序地看见
│   ├── 同一 proxy → acquire/release + 最小正确 scope
│   └── generic 与 async/tensormap 等 proxy 间 → 所需 fence.proxy + completion
├── 异步搬运还没到达
│   ├── cp.async → commit_group / wait_group
│   └── TMA → mbarrier transaction wait
└── Tensor 计算还没产出
    ├── mma.sync → 正常寄存器依赖
    ├── wgmma → wgmma wait_group
    └── tcgen05 → tcgen05 commit/fence/wait
```

## 13.7.4 Blackwell 代码审查清单

1. 是否将 execution barrier、memory ordering、async completion 写成了三件独立的事？
2. 每个 barrier 的参与者是否在所有控制流路径上匹配？
3. 每个 `mbarrier` 是否有清晰的 init、phase token、expect/complete transaction 与复用边界？
4. `cp.async`/WGMMA wait 是否被放在首次真实消费点，而不是紧跟 commit？
5. 是否在 async/generic/tensormap 或 TMEM 专用访问路径之间遗漏规范要求的 fence？
6. 是否明确标注目标是 `sm_100a`/`sm_103a` 还是 `sm_120`？
7. 每条声称的 SASS 映射是否由实际 Toolkit 和目标架构的反汇编验证？

## 13.7.5 推荐学习顺序

```text
1. CUDA Thread Hierarchy
        ↓
2. Execution Barrier
   warp → CTA → cluster → cooperative grid
        ↓
3. Memory Model
   order + scope + fence
        ↓
4. Atomic & Collective
   atom/red/CAS + shfl/vote/match/redux/elect
        ↓
5. Async Pipeline
   cp.async → mbarrier → TMA → WGMMA
        ↓
6. Proxy Memory Model
   generic / async / tensormap / alias
        ↓
7. Blackwell Cluster/TMEM Synchronization
   CTA pair + tcgen05 + required synchronization
```

为什么把 proxy 放在 async pipeline 之后：先通过真实数据通路理解“谁在访问同一块 memory”，再学习跨 proxy ordering，比先背 `fence.proxy.*` 更容易建立正确直觉。实际设计 pipeline 时，proxy fence 与 completion 仍需同时考虑，并不存在先后独立的运行阶段。

每一步都应把 CUDA 源码、PTX、SASS 和运行结果并排验证；进入 WGMMA/tcgen05 前先固定 Toolkit、`-arch` 和真实 GPU capability。

返回：[同步全景](./00-overview.md)。
