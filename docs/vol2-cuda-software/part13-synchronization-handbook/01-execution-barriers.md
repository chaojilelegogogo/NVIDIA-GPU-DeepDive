# 13.1 Execution Barrier：让哪些执行者会合

> Barrier 的第一问题永远是“哪些线程必须一起到达”。先选择最小正确范围，再讨论可见性和异步完成。

## 13.1.1 CTA barrier：`bar.sync`

| 字段              | 内容                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| 目的              | 让 CTA 中指定数量的线程会合后继续执行                                                              |
| 作用域 / 对象     | 一个 CTA；参与线程数必须与实际到达者一致                                                           |
| Execution barrier | 是                                                                                                 |
| Memory ordering   | 对 CTA 内相关 shared/global 访问提供 `__syncthreads()` 定义的可见性；不是任意 scope 的独立 fence |
| CUDA              | `__syncthreads()`；子组优先使用 Cooperative Groups                                               |
| PTX               | `bar.sync id{, count}`；新语法可见 `barrier.cta.sync`                                          |
| 典型 SASS         | `BAR.SYNC`（具体编码随架构变）                                                                   |
| 推荐              | 所有架构的 CTA producer/consumer 基线；避免放在可能分歧的路径                                      |

```cuda
__shared__ float tile[256];
tile[threadIdx.x] = input[i];
__syncthreads();               // 全部 CTA 线程必须参与
float x = tile[(threadIdx.x + 1) & 255];
```

在同一 CTA 内，一部分线程跳过 `__syncthreads()`、另一部分进入，会造成未定义行为，常见表现为死锁。若只需要一个 warp，会合范围应降为 `__syncwarp(mask)`。

### 13.1.1.1 CTA barrier 的四种行为

| 形式                                | 是否等待调用线程                    | 返回值                     | CUDA 常见入口                                               |
| ----------------------------------- | ----------------------------------- | -------------------------- | ----------------------------------------------------------- |
| `bar.sync` / `barrier.cta.sync` | 是                                  | 无                         | `__syncthreads()`                                         |
| `bar.arrive` / arrive 形式        | 否，只登记当前 warp/thread 集合到达 | 无                         | 通常通过低层 PTX；高层 split-phase 更常用 `cuda::barrier` |
| `bar.red.popc`                    | 是                                  | predicate 为真的参与线程数 | `__syncthreads_count(pred)`                               |
| `bar.red.and/or`                  | 是                                  | 全部为真 / 任一为真        | `__syncthreads_and/or(pred)`                              |

`bar.arrive` 适合生产者登记完成后继续执行独立工作，另一组参与者以匹配 barrier id/count 等待。它不是“发出事件后永远有效”：named CTA barrier 的 pending arrival、重用时机、参与计数必须配对，且 PTX 对某些 arrive 形式规定 warp-aligned participation。

```cuda
int active_count = __syncthreads_count(is_active); // 常见 bar.red.popc
int all_ready    = __syncthreads_and(ready);       // 常见 bar.red.and
int any_error    = __syncthreads_or(error);        // 常见 bar.red.or
```

`bar.red` 同时做 CTA 会合和 predicate reduction，不能用普通 `atomicAdd` 完全替代其参与协议；反过来，它只归约 predicate，不是任意数值 reduction。

## 13.1.2 Warp barrier：`bar.warp.sync`

| 字段              | 内容                                                           |
| ----------------- | -------------------------------------------------------------- |
| 目的              | 让 mask 指定的 warp lanes 会合                                 |
| 作用域 / 对象     | 单 warp；mask 中的非退出 lane 必须以兼容方式参与               |
| Execution barrier | 是                                                             |
| Memory ordering   | 仅为该 warp 的通信建立所需顺序；不扩大到 CTA                   |
| CUDA              | `__syncwarp(mask)`，默认 `0xffffffff` 仅适合全 warp 都活跃 |
| PTX               | `bar.warp.sync mask`                                         |
| 典型 SASS         | `WARPSYNC`                                                   |
| 推荐              | Volta+ 的 warp 内 shared-memory handoff 或分歧后重会合         |

Volta 的 Independent Thread Scheduling 后，warp 中不同 lane 可处于不同动态指令位置；“warp 天然 lockstep，所以无需同步”不再是正确通信协议。`__activemask()` 是**当前位置**活动 lane 的快照，不保证它是算法应使用的稳定参与集合；应在分歧前保存并传递所需 mask。

### 13.1.2.1 带 `.sync` 的 Warp collective

`shfl.sync`、`vote.sync`、`match.sync`、`redux.sync` 也带有参与 mask 和 convergence 契约，但应归类为“collective communication”，而不是 general-purpose warp barrier：

- `shfl.sync`：在参与 lane 的寄存器间交换数据；
- `vote.sync`：对 predicate 做 all/any/ballot；
- `match.sync`：找出值相等的参与 lane；
- `redux.sync`：对参与 lane 的整数做规约；
- `elect.sync`：选择唯一 leader。

它们只保证该 collective 所需的参与与寄存器结果，不等价于 CTA/global memory fence。完整列表见 [Collectives and Atomics](05-collectives-and-atomics.md)。

## 13.1.3 Cluster barrier：跨 CTA 的会合

> 执行层级与 `clusterDim`/`block_rank`：[Part 02 §2.2.4](../part02-cuda-programming-model.md)。DSM 地址与性能：[Part 03 §3.10](../part03-cuda-memory/08-cluster-dsm-tmem.md)。CG API：[Part 07 §7.4.5](../part07-cuda-cpp-api.md)。本节只回答 **execution barrier + DSM 可见性**。

| 字段              | 内容                                                                               |
| ----------------- | ---------------------------------------------------------------------------------- |
| 目的              | 同一 Thread Block Cluster 的 CTA 会合，保护 DSM 读写交接                           |
| 作用域 / 对象     | Hopper+ cluster；所有 cluster CTA 必须按协议参与                                   |
| Execution barrier | 是                                                                                 |
| Memory ordering   | 对 cluster 范围 DSM/shared 通信建立规定的可见性；不自动成为 grid/global 的通信协议 |
| CUDA              | `cooperative_groups::this_cluster().sync()`                                      |
| PTX               | cluster barrier 指令族；以目标 Toolkit PTX ISA 为准                                |
| SASS              | 依赖目标架构，使用 `nvdisasm` 核实                                               |
| 推荐              | TMA multicast/DSM 的阶段边界；仅在确有跨 CTA 数据复用时使用                        |

### 13.1.3.1 为什么 `cluster.sync()` 属于 Synchronization，不属于 Memory

| 问题                                 | 原语                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| 同 cluster 的 CTA 是否都到达交接点？ | `cluster.sync()` 【execution】                      |
| peer shared 地址如何得到？           | `map_shared_rank` 【DSM / Memory，Part 03】         |
| 异步 TMA 是否写完 remote shared？    | mbarrier transaction 【async completion，13.3】       |
| 普通 global flag 跨任意 CTA 发布？   | release/acquire +`.gpu`/`.sys` 【ordering，13.2】 |

`cluster.sync()` 不能让一个普通、未按 cluster 调度的 CTA 与另一个 CTA 通信。启动时必须声明并满足 cluster dimension。

### 13.1.3.2 DSM visibility 最小协议

```text
所有 CTA：初始化本地 shared
        ↓
cluster.sync()          ← 会合；此后可安全 map / 读 peer
        ↓
CTA_i：map_shared_rank → 读/写 peer shared
        ↓
cluster.sync()          ← 会合；此后 owner 才可复用/退出 shared
```

可见性保证（在规范定义的 cluster barrier 语义下）：

1. **第一次 sync 之后**：参与 CTA 此前对本 CTA shared 的写，对随后经 DSM 读取的 peer 可见（配合正确参与）；
2. **第二次 sync 之后**：peer 的 DSM 访问已完成，owner 才能销毁/复用该 shared；
3. sync **不会**等待 TMA/WGMMA 等异步引擎——异步路径仍要 mbarrier / wait_group。

配套 case：[`case03_cluster_sync.cu`](../../../src/part13-synchronization-handbook/01-execution-barriers/case03_cluster_sync.cu)（rank1 写本地 shared，`cluster.sync` 后 rank0 `map_shared_rank` 读到 11）。

### 13.1.3.3 Distributed barrier 与 mbarrier 的区别

所谓 distributed barrier，不是“任意 GPU 上分散的 CTA 都能同步”，而是 barrier state/地址可位于 cluster 中某个 CTA 的 shared memory，并通过 DSM/cluster address 被 peer CTA 操作。其生命周期受 owner CTA 和整个 cluster 约束。

| 对象                                                | 等待什么                               |
| --------------------------------------------------- | -------------------------------------- |
| `cluster.sync()` / cluster barrier                | 参与 CTA 的执行到达                    |
| `mbarrier`（可位于 shared，可被 remote complete） | arrival count + 可选 transaction bytes |
| TMA remote mbarrier completion                      | DMA 完成记入某个 CTA 的 barrier 对象   |

TMA remote completion ≠ 所有 cluster CTA 执行会合。需要全员会合时仍用 `cluster.sync()`（或等价协议）。

### 13.1.3.4 与 Memory Scope `.cluster` 的关系

Hopper 在 memory model 中增加了 **`.cluster` scope**（见 [13.2](02-memory-ordering.md)）：

```text
.cta → .cluster → .gpu → .sys
```

- `cluster.sync()`：execution barrier（谁到达）；
- `st.release.cluster` / `atom.*.cluster`：memory order+scope（写对谁可见）。

二者正交：DSM handoff 通常 **既要** cluster 会合，**也可**在需要时使用 cluster-scope atomic；不要用其中一个替代另一个。

## 13.1.4 Grid barrier：不是默认的跨 block 同步

| 字段 | 内容                                                                                                                                                                                                                                            |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 目的 | 全 cooperative grid(协作组里的所有block) 的 CTA 会合                                                                                                                                                                                            |
| CUDA | `cooperative_groups::this_grid().sync()`                                                                                                                                                                                                      |
| 前提 | cooperative launch、硬件可同时驻留整个 grid（简单讲就是block数量小于等于sm数量，所有block可以同时开始执行，<br />才可以同步等待。）<br />必须通过cudaLaunchCooperativeKernel或者cudaLaunchCooperativeKernelMultiDevice（多GPU版本）启动kernel。 |
| 替代 | 拆成两个 kernel；或使用明确的 global-memory atomic/fence 协议                                                                                                                                                                                   |
| 推荐 | 仅在必须保留同一 kernel 状态且满足驻留约束时使用                                                                                                                                                                                                |
| PTX  | 没有专门的PTX指令，由atomic、acquire/release、barrier通过counter计数组合实现                                                                                                                                                                    |

普通 kernel 的 block 可以分批调度。若等待尚未被调度的 block，软件实现的“grid barrier”可能永久死锁；因此 `cudaDeviceSynchronize()` 是 Host 等待 Device 的 API，不能当作 kernel 内 grid barrier。

## 13.1.5 Warpgroup 与 Tensor 协作

Hopper WGMMA 的 128 线程 warpgroup 对同一 `wgmma.mma_async` 必须以一致控制流协作发射。但它不是一个可以用来代替 `bar.sync` 的通用、独立“warpgroup barrier”API。Warpgroup 不是一种新的通用线程同步机制。它只是 Tensor Core 指令（WGMMA/tcgen05）要求的一种协作线程组织方式。

WGMMA 的数据/结果依赖由其 `fence`、`commit_group`、`wait_group` 协议管理，详见 [Tensor Synchronization](04-tensor-synchronization.md)。

Blackwell 的 `tcgen05` 允许不同角色线程组织生产/消费，但也不应凭“warpgroup”名称假设泛化的会合语义。需要 CTA 或 cluster 会合时，使用对应的 barrier；需要 Tensor 完成时，使用 Tensor 指令族的 wait。

## 13.1.6 选择顺序

先判断问题属于哪一类，再选原语。下面前两行是 **execution barrier**；后几行是其它类别，不要用 CTA barrier 替代：

```text
只需同一 warp 的 lanes 会合？       → __syncwarp(mask)          【execution】
同一 CTA 的 shared memory 会合？    → __syncthreads / cuda::barrier 【execution】
同 cluster 的 DSM 交接？           → cluster sync + memory/async 协议 【execution + 其它】
整个 grid 必须会合？                → cooperative grid.sync，或拆 kernel 【execution】
异步拷贝/TMA 的完成？               → cp.async wait / mbarrier wait  【async completion，见 13.3】
Tensor 指令结果尚未可用？           → wgmma/tcgen05 wait           【tensor completion，见 13.4】
跨线程写要被有序观察？              → release/acquire + scope      【memory ordering，见 13.2】
```

## 13.1.7 配套案例

- [`case01_cta_barrier_reduce.cu`](../../../src/part13-synchronization-handbook/01-execution-barriers/case01_cta_barrier_reduce.cu)：观察 `__syncthreads` 与 count/and/or 对应的 `bar.sync/bar.red`。
- [`case02_warp_sync.cu`](../../../src/part13-synchronization-handbook/01-execution-barriers/case02_warp_sync.cu)：观察 `__syncwarp` 的 `bar.warp.sync`；collective 通信案例放在 13.5。
- [`case03_cluster_sync.cu`](../../../src/part13-synchronization-handbook/01-execution-barriers/case03_cluster_sync.cu)：Hopper+ cluster launch、DSM 映射与 `cluster.sync()`。
- [`case04_grid_sync.cu`](../../../src/part13-synchronization-handbook/01-execution-barriers/case04_grid_sync.cu)：cooperative launch 与 `grid.sync()`；设备不支持时安全跳过。
- [`case05_bar_arrive.cu`](../../../src/part13-synchronization-handbook/01-execution-barriers/case05_bar_arrive.cu)：同一 CTA named barrier 上 `bar.arrive`（不阻塞）与 `bar.sync`（等待）配对。

下一篇：[Memory Ordering](02-memory-ordering.md)。
