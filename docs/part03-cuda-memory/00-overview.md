# 第三部分 CUDA Memory：从存储层次到 Blackwell TMA

> 本部分不把 CUDA Memory 当作 API 列表，而是追踪一条主线：**算力增长快于数据供给能力，因此每一代硬件都把更多“搬数据、算地址、处理边界、重排布局、通知完成”的工作从通用线程卸载到专用硬件。**

## 学习路线

```text
存储层次与数据路径
  → 合并访问、对齐、bank conflict
  → ld.global + st.shared（同步、寄存器中转）
  → cp.async（Ampere，小粒度、逐线程发起）
  → cp.async.bulk（Hopper，大块线性搬运）
  → TMA / cp.async.bulk.tensor（Hopper，多维描述符）
  → Blackwell 扩展（新类型、gather/scatter、wide im2col、CTA pair）
  → Cluster/DSM、TMEM 与完整 GEMM pipeline
```

## 文件导航

| 文件 | 核心问题 |
|---|---|
| [01-memory-hierarchy.md](./01-memory-hierarchy.md) | Global/Shared/Register/Cache/Unified/Pinned Memory 各自解决什么问题？ |
| [02-access-patterns.md](./02-access-patterns.md) | Coalescing、alignment、transaction、bank conflict、swizzle 如何统一理解？ |
| [03-data-movement-evolution.md](./03-data-movement-evolution.md) | 从线程搬运到 TMA 的因果链和选择标准是什么？ |
| [04-ampere-cp-async.md](./04-ampere-cp-async.md) | 非 bulk `cp.async` 的 PTX、C++ pipeline、对齐、group 语义如何使用？ |
| [05-hopper-bulk-and-tma.md](./05-hopper-bulk-and-tma.md) | `cp.async.bulk`、TensorMap、1D–5D TMA、mbarrier、swizzle、multicast 如何使用？ |
| [06-blackwell-tma.md](./06-blackwell-tma.md) | Blackwell 给 TMA 增加了哪些类型、模式、CTA-pair 与 bulk-reduce 能力？ |
| [07-pipeline-examples.md](./07-pipeline-examples.md) | cp.async/TMA 双缓冲、GEMM producer-consumer、写回协议怎么搭？ |
| [08-cluster-dsm-tmem.md](./08-cluster-dsm-tmem.md) | DSM 地址/性能、`map_shared_rank`、TMA multicast、TMEM 如何组成完整数据通路？（Cluster 执行模型见 Part 02） |
| [09-debug-performance.md](./09-debug-performance.md) | 如何判断正确性、检查生成指令、定位没有重叠或 bank conflict？ |

## 原节号兼容映射

全书其它部分已经大量使用“第三部分 3.x”引用。文件夹化后继续保留原节号语义；新增专题以子节形式插入，不重新解释旧编号：

| 原节号 | 主题 | 新位置 |
|---|---|---|
| 3.1 | 存储空间总览 | [01-memory-hierarchy.md](./01-memory-hierarchy.md) |
| 3.2–3.5 | Coalescing、Alignment、Bank Conflict、Transaction | [02-access-patterns.md](./02-access-patterns.md) |
| 3.6–3.7 | Tiled GEMM 动机、`ld.global + st.shared` | [03-data-movement-evolution.md](./03-data-movement-evolution.md) |
| 3.8 | Ampere `cp.async` | [04-ampere-cp-async.md](./04-ampere-cp-async.md) |
| 3.9 | Hopper bulk copy / TMA | [05-hopper-bulk-and-tma.md](./05-hopper-bulk-and-tma.md) |
| 3.9 扩展 | Blackwell TMA 指令与 TensorMap 模式 | [06-blackwell-tma.md](./06-blackwell-tma.md) |
| 3.8–3.9 实战 | cp.async/TMA pipeline | [07-pipeline-examples.md](./07-pipeline-examples.md) |
| 3.10–3.11 | Cluster/DSM、TMEM | [08-cluster-dsm-tmem.md](./08-cluster-dsm-tmem.md) |
| 3.12–3.13 | 跨代总结与诊断 | [本页](./00-overview.md)、[09-debug-performance.md](./09-debug-performance.md) |

相关深入阅读：[第二部分 Cluster 执行层级](../part02-cuda-programming-model.md)、[第七部分 `cluster_group` API](../part07-cuda-cpp-api.md)、[第四部分执行流水线](../part04-execution-pipeline.md)、[第五部分 Tensor Core 手册](../part05-tensor-core-handbook/00-overview.md)、[第十一部分性能优化](../part11-performance-optimization.md)、[第十三部分 Cluster barrier / 异步同步](../part13-synchronization-handbook/01-execution-barriers.md)。

## 先纠正四个常见混淆

1. **`cp.async` 不等于 TMA。** 非 bulk `cp.async` 从 Ampere 开始，通常每个参与线程搬 4/8/16B；TMA 属于 `cp.async.bulk.tensor` 指令族，以 TensorMap 描述多维区域。
2. **`cp.async.bulk` 不一定是 tensor copy。** 它也能只接收源地址、目标地址和字节数，做线性 bulk copy；有 `.tensor` 才由 TensorMap 生成多维地址。
3. **异步完成不等于线程会合。** `wait_group` 或 mbarrier transaction 完成只回答“数据到了没有”；多个线程能否安全消费，还要满足相应的 CTA/cluster 交接协议。
4. **“Blackwell TMA”不是一组所有 Blackwell 芯片都相同的能力。** PTX 目标需细分 `sm_100a`/`sm_103a`、同 family 的 `f` 目标、`sm_110*` 与消费级 `sm_120*`；某些 gather/scatter、sub-byte、swizzle atomicity、CTA-pair 变体受目标限制。

## 三种完成机制

```text
Ampere 非 bulk cp.async
  cp.async → commit_group → wait_group N

Hopper/Blackwell global → shared bulk/TMA load
  mbarrier arrive/expect transaction → cp.async.bulk(.tensor) → wait phase

Hopper/Blackwell shared → global bulk/TMA store
  cp.async.bulk(.tensor) ... bulk_group
    → cp.async.bulk.commit_group
    → cp.async.bulk.wait_group N
```

完成机制由**方向和指令变体**决定，不能把普通 `cp.async.commit_group`、bulk group 与 mbarrier 混用。

## 3.12 跨代总结

| 机制 | 发起方式 | 地址生成 | 寄存器中转 | 典型完成协议 |
|---|---|---|---|---|
| `ld.global + st.shared` | 每线程 | 软件逐线程 | 有 | 普通依赖 + CTA barrier |
| 非 bulk `cp.async` | 每线程小片段 | 软件逐线程 | 无 | `commit_group/wait_group` |
| `cp.async.bulk` | 少量线程/单 issuer | base + size | 无 | mbarrier 或 bulk group |
| TMA tensor copy | 单 issuer | TensorMap 硬件展开 | 无 | load:mbarrier；store:bulk group |
| Blackwell TMA 扩展 | 单 issuer/CTA pair | descriptor + mode | 无 | mbarrier/bulk group/CTA group |

## 架构与 PTX 版本意识

本文以 CUDA 13.x / PTX ISA 9.3 的公开文档为语义基线，同时在每个新能力旁标出引入版本或架构。PTX 是版本化虚拟 ISA：

- “PTX 文档中存在”不等于当前 `-arch` 能执行；
- `sm_100a` 中的 `a` 表示 architecture-specific，不能假定可前向兼容；
- CUDA C++ experimental wrapper、CUTLASS/CuTe 接口可能随 Toolkit 变化；
- 最终应同时核对 `nvcc --version`、生成 PTX、目标 capability 与 SASS。

## 推荐实操环境

| 内容 | 最低建议 |
|---|---|
| `cp.async` | Ampere `sm_80` |
| bulk copy / 基础 TMA | Hopper `sm_90` |
| TMA multicast | Hopper 的 cluster-capable 目标，优先 `sm_90a` |
| gather4、wide im2col、CTA pair、新 swizzle/sub-byte | 按 PTX target notes 使用数据中心 Blackwell |
| TMEM / `tcgen05` | 数据中心 Blackwell `sm_100a`/`sm_103a`，不适用于 `sm_120` |

## 参考规范

- [PTX ISA：Tensor-map](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-map)
- [PTX ISA：Asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy)
- [CUDA C++ Programming Guide：TMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
- [CUDA Driver API：Tensor Memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

下一篇：[存储层次与真实数据路径](./01-memory-hierarchy.md)。
