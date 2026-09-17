# 第十三部分 NVIDIA GPU 同步手册（Fermi → Blackwell）

> 本部分不按 CUDA API 罗列功能，而按**硬件同步原语**组织：谁必须停下并会合、哪一次写对谁何时可见、以及异步硬件何时完成，是三件不同的事。目标是读懂 CUDA C++、PTX、SASS 与 CUTLASS/CuTe 中的同步协议，而不是把所有问题归结为 `__syncthreads()`。

## 13.0 阅读方式与版本边界

本手册覆盖 Fermi 的传统 CTA 屏障，到 Ampere `cp.async`、Hopper TMA/WGMMA/Cluster、以及**数据中心** Blackwell（`sm_100a`/`sm_103a`）的 `tcgen05`。PTX 是虚拟 ISA，SASS 是由 `ptxas` 针对具体目标生成的机器码；表中的 SASS 只能作为常见反汇编线索，必须以目标 CUDA Toolkit 和 `nvdisasm` 的实际输出为准。

消费级 Blackwell `sm_120` 不具备 TMEM/`tcgen05`；不要把数据中心 Blackwell 的同步协议迁移到 RTX 50 系列。

## 13.1 四个正交类别

```text
GPU Synchronization & Memory Model
├── 1. Thread Execution Synchronization：谁必须到达，谁才能继续
│   ├── Warp：__syncwarp / bar.warp.sync
│   ├── CTA：__syncthreads / bar.sync / bar.arrive / bar.red
│   ├── Cluster：cluster.sync / barrier.cluster / DSM handoff
│   └── Grid：cooperative grid.sync
│
├── 2. Memory Ordering & Visibility：访问以什么顺序被谁观察
│   ├── Order：relaxed / acquire / release / acq_rel / seq_cst
│   ├── Scope：thread / cta / cluster / gpu / sys
│   ├── Fence：membar / fence.sc / fence.proxy.*
│   └── Proxy：generic / async / tensormap / alias / fabric
│
├── 3. Asynchronous Operation Synchronization：异步工作何时完成
│   ├── cp.async：commit_group / wait_group
│   ├── mbarrier：arrival / phase / transaction bytes / wait
│   ├── TMA：cp.async.bulk.tensor + mbarrier completion
│   ├── WGMMA：fence / commit_group / wait_group
│   └── tcgen05：TMEM + commit/fence/wait
│
└── 4. Atomic & Collective Coordination：原子更新和局部协同
    ├── atom / red / CAS / scoped ordered atomic
    ├── red.async / multimem.red
    └── activemask / shfl.sync / vote.sync / match.sync / redux.sync / elect.sync
```

这四类会在一个真实 pipeline 中组合，但不能互相替代：

1. execution barrier 等待参与线程，不代表独立异步引擎已经完成；
2. fence/order 规定观察顺序，不会把其它线程叫到同一程序位置；
3. async completion 等待 copy/Tensor 工作，不自动让整个 CTA/cluster 会合；
4. atomic 保证目标更新不可撕裂，collective 保证参与 mask 的协作；二者都不自动发布其它 payload。

### 13.1.1 两个容易分类错误的边界

公开 PTX 没有一个可把所有 Tensor 操作统一归类的“tensor proxy”：

- 普通 `ld/st/atom` 属于 generic proxy；
- 非 bulk Ampere `cp.async` 在当前 PTX memory model 中也是 weak generic-proxy operation；
- TMA/`cp.async.bulk` 的数据传输属于 async proxy；
- TMA descriptor 通过 tensormap proxy 访问；
- WGMMA 明确使用 async proxy，并另有专用 `wgmma.fence/commit/wait`；
- `tcgen05`/TMEM 按该指令族的 required synchronization 推理，不能凭名称虚构通用 proxy。

同样，公开 TMA PTX 入口是 `cp.async.bulk.tensor.*`，而不是 `tma.async.load`。后者可以作为硬件行为的口语描述，但不能写进 PTX 映射表。

### 13.1.2 Memory Model 的三维坐标

```text
一次发布/获取是否正确？
├── Order：release 发布了什么，acquire 成功取得后约束什么？
├── Scope：producer 与 consumer 是否都被 .cta/.cluster/.gpu/.sys 覆盖？
└── Proxy：双方是否以相同访问方法触碰对象？若不同，是否需要 proxy fence？
```

Order、Scope、Proxy 的完整取值、合法组合和代码示例见 [Memory Ordering](02-memory-ordering.md)。Warp 是 execution/collective 的参与集合，不是 PTX 通用 memory scope。

### 13.1.3 Producer/consumer：为什么只写 `__threadfence()` 不够

目标是让 Block 1 只有在看到 `flag==1` 后才读取到 Block 0 写好的 `x==1`：

```text
Block 0 (producer)                 Block 1 (consumer)
x = 1;                             while (flag.load(acquire) == 0) {}
flag.store(release, 1);            use(x);
```

release flag store 保证此前对 `x` 的写在该发布之前可被观察；acquire flag load 保证随后读 `x` 不会在观察上越过取得该 flag。两端还必须选择覆盖两个 Block 的 scope，例如 device/GPU scope。

```cpp
// 示意：实际类型、地址空间和库版本应按项目 CUDA Toolkit 核对。
// producer
x = 1;
cuda::atomic_ref<int, cuda::thread_scope_device> f(flag);
f.store(1, cuda::memory_order_release);

// consumer
cuda::atomic_ref<int, cuda::thread_scope_device> f(flag);
while (f.load(cuda::memory_order_acquire) == 0) {}
int observed = x;
```

旧式 `x = 1; __threadfence(); flag = 1;` 只描述 producer 一侧的排序，**不是完整的跨 CTA 发布协议**：consumer 仍需一个匹配的 acquire/观察机制，且 flag 的读写必须避免普通非原子数据竞争。`__threadfence()` 不会让 Block 1 被唤醒，也不保证 TMA/WGMMA 等异步操作已经完成。

### 13.1.4 Order 与 Scope 的最小化原则

| 需求                    | order                    | scope      | 典型选择                           |
| ----------------------- | ------------------------ | ---------- | ---------------------------------- |
| CTA 内 shared handoff   | barrier 提供的规定可见性 | CTA        | `__syncthreads()`                |
| CTA 内 flag 发布        | release/acquire          | CTA        | 适用的 CTA-scope atomic/fence 协议 |
| 两个 CTA 间 global 发布 | release/acquire          | GPU        | device-scope atomic 发布/获取      |
| GPU 通知 CPU/peer       | release/acquire          | system     | system-scope atomic/fence 协议     |
| generic ↔ async 交接   | 顺序 + completion        | 所需 scope | proxy fence +`mbarrier`/wait     |

选择大于所需的 scope 会增加成本；小于实际 producer/consumer 集合则不正确。execution barrier、memory order/scope 和 async completion 仍是三个不同问题。

## 13.2 Object-state synchronization

Hopper/Blackwell 的关键变化不是“线程不再同步”，而是异步 pipeline 新增了可复用的**对象状态**：

| 状态 | 回答的问题 | 典型对象 |
|---|---|---|
| arrival count | 还有多少参与者未到达当前 phase？ | `mbarrier` |
| phase/parity/token | 当前等待的是 barrier 的哪一代复用？ | `mbarrier` / `cuda::barrier` |
| transaction bytes | 还有多少异步数据未完成？ | TMA + `expect_tx/complete_tx` |
| async group | 哪批 copy/Tensor 指令已提交、允许多少批仍未完成？ | `cp.async`、WGMMA |
| stage ownership | buffer 现在属于 producer 还是 consumer？ | `cuda::pipeline` |

因此 `mbarrier.wait` 不是传统 `__syncthreads()` 的新拼写：前者等待某个对象 phase 的 arrival/transaction 条件，后者让整个 CTA 会合。完整状态机见 [Async Pipeline](03-async-pipelines.md)。

## 13.3 统一原语模板

后续每项使用以下字段：

| 字段                   | 含义                                                   |
| ---------------------- | ------------------------------------------------------ |
| 目的                   | 要解决的依赖，而非助记符字面意思                       |
| 参与者 / scope / proxy | execution 参与者、memory scope、访问代理分别是什么     |
| Execution barrier      | 是否让发起者/参与者等待                                |
| Memory ordering        | relaxed/acquire/release/acq_rel/sc，以及哪些访问被排序 |
| CUDA / PTX / SASS      | 上层入口、规范语义、典型反汇编线索                     |
| 协议                   | 必要的 arrive/commit/wait/fence 顺序                   |
| 演进与推荐             | 最低架构、替代方式、Blackwell 建议                     |

## 13.4 演进时间线

| 时代                     | 关键原语                                              | 同步模型的变化                                                |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------- |
| Fermi → Pascal          | `bar.sync`、`membar`、atomics、warp lockstep 假设 | CTA 会合和显式内存栅栏为主                                    |
| Volta                    | `*.sync` warp collectives                           | Independent Thread Scheduling 取消“warp 天然同步”的安全假设 |
| Ampere `sm_80`         | `cp.async` group、`mbarrier`                      | 数据搬运可在普通线程继续执行时完成                            |
| Hopper `sm_90`         | TMA transaction、cluster、WGMMA                       | 异步 DMA、跨 SM shared memory 与 Tensor 依赖进入协议中心      |
| Blackwell DC `sm_100a` | TMEM、`tcgen05`                                     | Tensor 完成、TMEM 可见性与 CTA-pair 协作成为一等问题          |

## 13.5 官方语义与实测边界

语义以 CUDA C++ Programming Guide、CUDA C++ Core Compute Libraries（`cuda::barrier`/`cuda::pipeline`）和随 Toolkit 发布的 PTX ISA 为准。建议对每个目标架构保存：

```bash
nvcc -arch=sm_80 -cubin kernel.cu -o kernel-sm80.cubin
nvcc -arch=sm_90a -cubin kernel.cu -o kernel-sm90a.cubin
nvdisasm -c kernel-sm90a.cubin
```

用此流程确认 SASS 助记符和控制依赖；不要从 SASS 的 scoreboard barrier 编号推导 CUDA `bar.sync` 语义。二者名称相似，但前者是调度器的指令依赖标记，后者是程序可见的线程会合原语。

## 13.6 配套实验与文档映射

| 文档章节 | Case 目录 | 覆盖要点 |
|---|---|---|
| 13.1 Execution | [`01-execution-barriers/`](../../../src/part13-synchronization-handbook/01-execution-barriers) | `bar.sync`、`bar.arrive`、`bar.red.*`、`bar.warp.sync`、cluster/grid |
| 13.2 Ordering | [`02-memory-ordering/`](../../../src/part13-synchronization-handbook/02-memory-ordering) | order、scope、fence.sc、`__threadfence`、Order×Scope payload |
| 13.3 Async | [`03-async-pipelines/`](../../../src/part13-synchronization-handbook/03-async-pipelines) | cp.async group、mbarrier phase/tx、TMA transaction、pipeline stage |
| 13.4 Tensor | [`04-tensor-synchronization/`](../../../src/part13-synchronization-handbook/04-tensor-synchronization) | `mma.sync` 基线；WGMMA/tcgen05 按目标限制 |
| 13.5 Collective/Atomic | [`05-collectives-and-atomics/`](../../../src/part13-synchronization-handbook/05-collectives-and-atomics) | `shfl/vote/match.sync`、atom vs red、ordered atomic 复用 13.2 |

总入口：[`src/part13-synchronization-handbook/README.md`](../../../src/part13-synchronization-handbook/README.md)。

### 13.6.1 Cluster / DSM 交叉主题（不要单独拆章）

Hopper Cluster 是横切主题，应按问题类型阅读：

| 问题 | 章节 |
|---|---|
| 为什么需要 Cluster？执行层级？ | [Part 02 §2.2.4](../part02-cuda-programming-model.md) |
| DSM 是什么、怎么寻址、性能？ | [Part 03 §3.10](../part03-cuda-memory/08-cluster-dsm-tmem.md) |
| `this_cluster` / `map_shared_rank` API？ | [Part 07 §7.4.5](../part07-cuda-cpp-api.md) |
| `cluster.sync()` / DSM 可见性？ | [本部分 13.1.3](01-execution-barriers.md) |
| `.cluster` memory scope？ | [本部分 13.2](02-memory-ordering.md) |

下一篇：[Execution Barrier](01-execution-barriers.md)。
