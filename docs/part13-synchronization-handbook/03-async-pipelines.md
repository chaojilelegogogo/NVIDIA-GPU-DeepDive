# 13.3 Async Pipeline Synchronization：完成不是会合

> 异步流水线要同时回答三件事：请求是否已提交、DMA 是否已完成、消费者是否已被允许使用结果。`commit`、`wait` 和 barrier/fence 不可互相替代。

## 13.3.1 Ampere `cp.async` group

| 字段 | 内容 |
|---|---|
| 目的 | 将 global → shared 的小块复制从寄存器中转路径移到异步硬件路径 |
| 最低架构 | Ampere `sm_80` |
| Execution barrier | 否；发起后线程可继续。`wait_group` 会等待调用线程的未完成 group |
| Memory ordering | completion 后仅表示复制到达；与其它线程交接时仍需正确 CTA 协议 |
| CUDA | `cuda::memcpy_async`、`cuda::pipeline`、部分 Cooperative Groups 封装 |
| PTX | `cp.async.*` → `cp.async.commit_group` → `cp.async.wait_group N` |
| 典型 SASS | 常见为 `LDGSTS`，以实测为准 |

```ptx
cp.async.cg.shared.global [dst0], [src0], 16;
cp.async.cg.shared.global [dst1], [src1], 16;
cp.async.commit_group;
// 可以在这里计算上一 stage
cp.async.wait_group 0;     // 仅当没有未完成 group 时继续
```

`wait_group N` 的含义是允许最多 N 个较早 group 仍未完成，并不是“等待第 N 个 group”。双/三缓冲通常故意等待 `N > 0`，以保持下一 stage 的搬运与当前 stage 计算重叠。warp 中的参与线程必须以一致协议发起、提交和等待；不要在不匹配的分歧路径中只让部分 lane 改变 group 状态。

非 bulk `cp.async` 虽然执行上异步，但当前 PTX memory model 将其定义为 weak generic-proxy operation；不要因为名字含 `.async` 就把它与 Hopper bulk/TMA 的 async proxy 混为一类。

## 13.3.2 `mbarrier`：可复用的 phase + transaction 对象

`mbarrier` 位于 shared memory，状态包含到达计数、phase 与（支持异步 transaction 时）尚未完成的 transaction 字节。它不是普通 `bar.sync` 的同义词。

| 操作 | 作用 |
|---|---|
| `mbarrier.init` | 初始化预期 arrival 数；初始化阶段须由正确的 CTA 协议保护 |
| `mbarrier.arrive` / `arrive_drop` | 抵达当前 phase；`drop` 同时减少后续 phase 参与数 |
| `mbarrier.arrive.expect_tx` | 抵达并声明期望的异步 transaction 字节 |
| `mbarrier.complete_tx` | 完成指定 transaction 字节（由适用的生产者/硬件路径执行） |
| `mbarrier.test_wait` / `try_wait` | 依据 phase token 检查或等待完成 |
| `mbarrier.inval` | 生命周期结束后失效，之后才可复用该存储 |

```text
init(expected arrivals)
  ↓
producer: arrive.expect_tx(bytes) ──→ async copy/TMA
  ↓                                  ↓
arrival count reaches zero       transaction bytes complete
  └─────────────── both true ────────┘
                  phase flips
                      ↓
consumer: try_wait(token) → consume → 下一 phase
```

等待方必须保存与本次 arrival 对应的 phase token；只轮询“看起来已完成”的旧 phase 会误把复用后的对象状态当成当前数据。`mbarrier` 的 arrival 与 transaction completion 必须都满足，少写 `expect_tx` 或错误的字节数会使等待过早/永久无法满足。

### 13.3.2.1 Object lifecycle

```text
shared storage 分配
  → mbarrier.init(expected arrivals)
  → 让初始化对所有参与者/所需 proxy 可见
  → arrive / expect_tx / async work
  → test_wait 或 try_wait 当前 token/parity
  → phase 完成并翻转
  → 下一轮以新 phase 复用
  → 所有使用者结束后 inval/析构
```

`test_wait` 是测试型操作；`try_wait` 可带 suspend hint/循环等待形式，具体签名随 PTX 版本变化。两者都必须测试本轮 state，而不是把一次成功结果永久缓存。`arrive_drop` 会减少未来 phase 的预期参与者数，不只是本轮“少 arrive 一次”。

## 13.3.3 Hopper TMA：以 transaction 完成通知 DMA

| 字段 | 内容 |
|---|---|
| 目的 | 单线程描述并发起多维 tile 的 global ↔ shared 拷贝 |
| 最低架构 | Hopper `sm_90` |
| CUDA | `CUtensorMap`（Driver API）及 CuTe/CUTLASS 封装；没有一对一通用 Runtime intrinsic |
| PTX | `cp.async.bulk.tensor...`，配合 `mbarrier` transaction |
| completion | TMA 将指定字节的完成记入 mbarrier；consumer 以 phase/token 等待 |
| Cluster | 可写入 `.shared::cluster` 并 multicast；每个目标 CTA 的生命周期仍需正确管理 |
| Proxy | bulk tensor 数据传输使用 async proxy；TensorMap descriptor 使用 tensormap proxy |

最小协议是：初始化 mbarrier → 在发起前让 barrier 知道预期 transaction → 发起 TMA → consumer 等待相同 phase 完成 → 消费 shared tile → phase 复用。TMA 完成不等于所有 CTA 已会合；若所有 cluster CTA 都要访问彼此 DSM，另需 cluster 范围的执行同步或等价的明确协议。

公开 PTX 指令名是 `cp.async.bulk.tensor.{1d..5d}...`，不是 `tma.async.load`。文档中“TMA load”描述硬件行为时可以简写，但做 CUDA→PTX 映射时必须写真实 mnemonic。

## 13.3.4 `cuda::barrier` 与 `cuda::pipeline`

这些 C++ 抽象将 producer/consumer 协议包装为 arrival token、stage acquire/commit 和 consumer wait/release。它们有助于减少手写 phase 错误，但不抹掉底层限制：

```cuda
// 示意：每个 stage 的 acquire/commit 必须与 consumer wait/release 配对
pipe.producer_acquire();
cuda::memcpy_async(block, shared_tile, global_tile, bytes, pipe);
pipe.producer_commit();

pipe.consumer_wait();
consume(shared_tile);
pipe.consumer_release();
```

是否生成 `cp.async`、以及是否能走硬件加速路径，取决于架构、地址空间、对齐和编译器可证明的条件。用 `nvdisasm` 验证，不要只凭 API 名称假定生成了异步 copy。

### 13.3.4.1 Producer/consumer stage ownership

```text
Producer warp                         Consumer warpgroup
acquire EMPTY stage
  → TMA/cp.async 写 shared
  → commit / arrive full
                                      wait FULL stage
                                        → WGMMA/tcgen05 consume
                                        → release EMPTY stage
```

`cuda::pipeline` 的 `producer_acquire/commit` 与 `consumer_wait/release` 管理的是 buffer stage 所有权；底层可以是 `cp.async` group、mbarrier 或目标相关实现。Warp specialization 只是把角色分给不同 warp，不会自动建立任何 handoff。

一个完整 Hopper GEMM stage 至少同时存在：

1. TMA 的 mbarrier transaction completion；
2. generic/async proxy 交接；
3. consumer warpgroup 的 WGMMA fence/commit/wait；
4. stage empty/full 的 producer-consumer 所有权。

漏掉其中之一可能得到“偶尔正确”的 race；把所有位置都换成 `__syncthreads()` 则既不完整也会破坏流水重叠。

## 13.3.5 常见错误

| 症状 | 通常原因 | 修复方向 |
|---|---|---|
| 首个 stage 正确、循环后出错 | 忘记 phase/token 或过早复用 barrier | 把每 stage 的 token 与 release/inval 生命周期写清楚 |
| 性能没有重叠 | `wait_group 0` 紧跟 commit | 增加 stage，延后 wait 到真正消费点 |
| 消费者读到旧 shared 数据 | 只等线程 arrival，未等 transaction | 使用 `expect_tx` + completion wait |
| cluster 偶发错误 | 只同步本 CTA，忽略 DSM 读写者 | 采用 cluster 同步或完整的跨 CTA 协议 |

## 13.3.6 配套案例

- [`case01_cp_async_pipeline.cu`](../../src/part13-synchronization-handbook/03-async-pipelines/case01_cp_async_pipeline.cu)：`cuda::memcpy_async`/pipeline 与 `cp.async.commit_group/wait_group`。
- [`case02_mbarrier_phases.cu`](../../src/part13-synchronization-handbook/03-async-pipelines/case02_mbarrier_phases.cu)：可复用 `cuda::barrier` 的 token/phase。
- [`case03_tma_transaction.cu`](../../src/part13-synchronization-handbook/03-async-pipelines/case03_tma_transaction.cu)：Hopper+ 1D TMA、`barrier_arrive_tx` 与 transaction bytes；不支持时跳过。

下一篇：[Tensor Synchronization](./04-tensor-synchronization.md)。
