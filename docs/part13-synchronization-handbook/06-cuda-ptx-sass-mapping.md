# 13.6 CUDA → PTX → SASS：同步映射速查

> CUDA API、PTX 与 SASS 不是一一对应关系。编译器可能内联、融合、删除或选择目标相关的低层序列。本表的“典型 SASS”用于反汇编定位，绝不是跨架构 ABI 承诺。

## 13.6.1 Execution barrier

| CUDA / 抽象 | PTX（语义入口） | 典型 SASS 线索 | scope | 最低架构 / 备注 |
|---|---|---|---|---|
| `__syncthreads()` | `bar.sync` / `barrier.cta.sync` | `BAR.SYNC` | CTA | 所有 CUDA 架构 |
| `__syncthreads_count/and/or` | `bar.red.popc/and/or` | `BAR.RED` / target-specific | CTA | 会合 + predicate reduction |
| 低层 split arrive | `bar.arrive` / `barrier.cta.arrive` 变体 | `BAR.ARRIVE` / target-specific | CTA | 登记到达但不等待；须匹配 barrier 协议 |
| `__syncwarp(mask)` | `bar.warp.sync` | `WARPSYNC` | warp | Volta+ 显式使用尤重要 |
| `cuda::barrier` | 常映射到 mbarrier 或传统 barrier，取决于实现 | `MBARRIER.*` / `BAR.*` | CTA | 需要查看生成物 |
| `cg::this_cluster().sync()` | cluster barrier 指令族 | target-specific | cluster | Hopper+；需要 cluster launch |
| `cg::this_grid().sync()` | cooperative-grid 机制 | target-specific | grid | cooperative launch；非普通 block barrier |

## 13.6.2 Memory ordering 与异步拷贝

| CUDA / 抽象 | PTX | 典型 SASS 线索 | 语义重点 |
|---|---|---|---|
| `__threadfence_block()` | `membar.cta` 或现代 `fence` 序列 | `MEMBAR` / target-specific | 只排序；不等待 |
| `__threadfence()` | `membar.gl` 或 `.gpu` 级 `fence` 序列 | `MEMBAR` / target-specific | device 范围发布协议 |
| `__threadfence_system()` | `membar.sys` 或 `.sys` fence 序列 | `MEMBAR` / target-specific | system 范围，代价更高 |
| 无一对一 CUDA API | `fence.proxy.*` / `fence.proxy.async` | target-specific | generic/async/tensormap 等 proxy 交接；无统一 tensor proxy |
| `cuda::memcpy_async` / `cuda::pipeline` | `cp.async.*` + commit/wait | 常见 `LDGSTS` | Ampere+；需满足代码生成条件 |
| TMA（CuTe/CUTLASS/inline PTX） | `cp.async.bulk.tensor.*` + `mbarrier` | target-specific | Hopper+ DMA transaction |
| `cuda::barrier` transaction 用法 | `mbarrier.*` | `MBARRIER.*` | phase、arrival、transaction completion |

## 13.6.3 Tensor completion

| CUDA / 抽象 | PTX | 典型 SASS 线索 | 协作范围 | 架构 |
|---|---|---|---|---|
| WMMA `mma_sync` | `mma.sync` | `HMMA` 等 | warp | Volta+ |
| 无稳定高层一对一 API | `wgmma.mma_async` + fence/commit/wait | `HGMMA` / `QGMMA` 常见 | warpgroup | Hopper |
| CuTe/CUTLASS 或 inline PTX | `tcgen05.*` + fence/commit/wait | `TCGEN05.*` / target-specific | 指令变体定义 | 数据中心 Blackwell |
| 消费级 FP4/FP6 路径 | `mma.sync.aligned.block_scale` 等 | target-specific | warp | `sm_120`；不是 tcgen05 |

## 13.6.4 Warp collective 与原子

| CUDA | PTX | 常见 SASS 线索 | 备注 |
|---|---|---|---|
| `__shfl*_sync` | `shfl.sync.*` | `SHFL` | 寄存器通信 + mask participation |
| `__ballot_sync` / `__any_sync` / `__all_sync` | `vote.sync.*` | `VOTE` | 使用显式参与 mask |
| `__match_any_sync` / `__match_all_sync` | `match.sync.*` | `MATCH` | Volta+ |
| 无稳定一对一 API | `redux.sync.*` | `REDUX` | warp 规约 |
| 无稳定一对一 API | `elect.sync` | target-specific | 不保证 leader 为 lane 0 |
| `atomic*` | `atom.*` | `ATOM` | 需要旧值 |
| 无返回 atomic 的编译选择 | `red.*` | `RED` | 无旧值时常可用 |
| 特殊路径 | `red.async.*` / `multimem.red.*` | target-specific | 严格核对 Toolkit/架构 |

ordered/scoped atomic 还应拆读 qualifier，例如：

```text
CUDA atomic_ref.fetch_add(relaxed), thread_scope_device
  → atom.add.relaxed.gpu

CUDA atomic_ref.store(release), thread_scope_system
  → st.release.sys 或目标相关等价序列
```

同一 CUDA 语义在 SASS 中可能由 atomic/load/store 的 STRONG scope 变体与 `MEMBAR` 组合实现，不能要求 PTX qualifier 与单条 SASS 一一同名。

## 13.6.5 建立可复现映射

建议每个表项附带一个最小 kernel，固定 CUDA Toolkit 与 `-arch`，并存档 PTX 和 SASS：

```bash
nvcc -std=c++20 -arch=sm_80 -ptx sync_probe.cu -o sync-sm80.ptx
nvcc -std=c++20 -arch=sm_90a -cubin sync_probe.cu -o sync-sm90a.cubin
nvdisasm -c sync-sm90a.cubin > sync-sm90a.sass
```

审阅时区分三层证据：

1. **语义**：CUDA/PTX 官方规范；
2. **代码生成**：固定 Toolkit、编译选项和目标下的 PTX/SASS；
3. **微架构推断**：benchmark、控制码或反编译观察，不能升级为语言保证。

下一篇：[Blackwell Synchronization Summary](./07-blackwell-summary.md)。
