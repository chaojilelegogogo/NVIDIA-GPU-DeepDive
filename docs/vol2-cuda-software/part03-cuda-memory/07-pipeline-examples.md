# 3.P cp.async/TMA 流水线实战：从正确到高性能

## 3.P.1 Pipeline 的三个状态

每个 stage 都应明确：

```text
EMPTY   ：buffer 可被 producer 覆盖
LOADING ：异步 copy 已发起，尚未完成
READY   ：copy 完成，可被 consumer 读取
```

consumer 用完后才把 READY 释放回 EMPTY。异步 API 的本质是实现这个状态机，而不是“调用一个 async 函数就自动重叠”。

## 3.P.2 非 bulk `cp.async` 双缓冲

概念时序：

```text
prologue:
  issue tile 0 → stage 0

steady state k:
  issue tile k+1 → stage (k+1)%2
  wait tile k
  CTA handoff
  compute tile k
  release stage k

epilogue:
  drain 最后一块
```

PTX 骨架：

```ptx
// prologue
cp.async.cg.shared.global [s0], [g0], 16;
...
cp.async.commit_group;

// steady
cp.async.cg.shared.global [s1], [g1], 16;
...
cp.async.commit_group;

cp.async.wait_group 1; // 保留一定 in-flight 深度
bar.sync 0;            // 若 CTA 跨线程消费
// compute s0
```

具体 `wait_group N` 要根据 pending group 数推导，不能机械复制 `1`。

## 3.P.3 TMA 双缓冲

每个 stage 需要：

- 128B 对齐的 shared tile；
- 一个可复用 mbarrier/`cuda::barrier` state；
- phase/token；
- EMPTY/READY 交接。

```text
stage 0: [barrier phase p0] [shared tile 0]
stage 1: [barrier phase p1] [shared tile 1]
```

伪代码：

```cuda
initialize_all_stage_barriers();

issue_tma(stage=0, coord=0, expected_bytes=TILE_BYTES);

for (int k = 0; k < K_TILES; ++k) {
  int read = k % STAGES;
  int write = (k + 1) % STAGES;

  wait_ready(read);

  if (k + 1 < K_TILES) {
    wait_empty(write); // 不能覆盖尚未消费的 stage
    issue_tma(write, k + 1, TILE_BYTES);
  }

  mma(shared_tile[read]);
  release_empty(read);
}
```

真实高性能实现常先 issue next 再 wait/compute current，并采用独立 producer warp；顺序取决于 barrier state machine 和资源依赖。

## 3.P.4 Warp specialization

Hopper/Blackwell 常将 CTA warp 分工：

```text
Producer warp：
  计算 tile coordinate
  发 TMA
  更新 full barrier

Consumer warpgroup：
  等 full
  WGMMA/tcgen05
  更新 empty barrier

Epilogue warp：
  等 accumulator
  写回
```

TMA 只需单线程发起，但通常保留一个 producer warp，原因是：

- scheduler 以 warp 为调度单位；
- producer 还需维护循环、descriptor coordinate、barrier；
- producer 可通过 `setmaxnreg` 等机制减少自身 register、让 consumer 使用更多 register；
- 单个 lane 必须位于及时获得调度的 converged warp 中。

## 3.P.5 Hopper GEMM 数据通路

```text
TMA load A/B
   ↓ mbarrier full
Shared A/B (swizzled)
   ↓ fence/operand ready
WGMMA async
   ↓ wgmma commit/wait
Register accumulator
   ↓ epilogue
Shared output
   ↓ async proxy fence + CTA handoff
TMA store
   ↓ bulk-group read completion
stage 可复用
```

这里至少有三套 completion：

1. TMA load 的 mbarrier transaction；
2. WGMMA 的 commit/wait group；
3. TMA store 的 bulk-group completion。

`__syncthreads()` 不能替代其中任意一个专用 completion。

## 3.P.6 Blackwell GEMM 数据通路

```text
TMA load / CTA pair load A/B
  ↓
Shared operand + scale
  ↓ async-proxy/Tensor ordering
tcgen05.mma / tcgen05.cp
  ↓
TMEM accumulator
  ↓ tcgen05 wait
tcgen05.ld → register epilogue
  ↓
shared/global output
```

Blackwell 把 accumulator 从 register 移到 TMEM，但没有消除 TMA stage。新的优化问题是：

- TMA producer 能否持续喂 shared；
- `tcgen05` 能否持续消费；
- TMEM column allocation 是否限制并发；
- CTA pair 的两个 CTA 是否在同一 phase；
- epilogue 读 TMEM 是否反过来堵塞主循环。

## 3.P.7 TMA store buffer 何时可覆盖

TMA store 是从 shared 读取、异步写 global。producer 最关心的是“copy engine 是否已经读完 shared”：

```cuda
cp_async_bulk_tensor_shared_to_global(...);
cp_async_bulk_commit_group();
cp_async_bulk_wait_group_read<0>();
// 到这里 shared source 才可安全重用
```

这不一定意味着其它 device/system observer 已按你需要的内存顺序看到 global 数据。跨 CTA producer-consumer、同 kernel 轮询 global 等场景还需要相应 release/acquire 协议。

## 3.P.8 Multicast pipeline

Cluster 中多个 CTA 共用 A tile 时：

```text
一个 CTA/issuer 发 multicast TMA
  → CTA mask 中每个目标 shared buffer
  → 每个目标对应 transaction completion
  → 各 CTA 消费
```

必须设计：

- 谁负责发起；
- mbarrier 放在哪个 CTA；
- destination 地址如何映射；
- 每个 CTA 何时 release；
- leader CTA 退出前 peer 是否仍依赖其 shared/barrier。

只画一个“broadcast 箭头”而不写生命周期，通常会留下 race。

## 3.P.9 Stage 数量如何选

近似条件：

```text
stage_count × compute_time_per_tile ≥ memory_latency
```

但 stage 增多同时增加：

- shared memory；
- barrier/state；
- descriptor/coordinate bookkeeping；
- pipeline drain 开销；
- 可能降低 occupancy。

调优步骤：

1. 先写 1-stage 正确版本；
2. 增至 2-stage 验证真正重叠；
3. 比较 3/4 stage；
4. 同时记录 occupancy、shared bytes、register、eligible warps、memory stall；
5. 选择端到端吞吐最优，而不是 in-flight stage 最多。

## 3.P.10 尾块策略

| 机制 | 尾块处理 |
|---|---|
| 普通 ld/st | predicate + 中性元写 shared |
| `cp.async` | `src-size` zero-fill 或 predicate |
| TMA tiled | TensorMap OOB zero/NaN fill |
| linear bulk `.ignore_oob` | 仅允许首尾少量 OOB，目标字节不确定，不等价 zero-fill |
| TMA store | 越界 destination 元素丢弃，起始坐标限制更严格 |

GEMM K 尾块的填充值通常为 0；max-reduction 的中性元可能是 `-inf`，不能直接使用 zero-fill 而不修正。

## 3.P.11 一个可执行的学习阶梯

1. `cp.async` vector add staging：确认生成 `LDGSTS` 类路径；
2. `cp.async` 两 stage tiled transpose：观察 bank conflict；
3. TMA 1D roundtrip；
4. TMA 2D OOB zero-fill；
5. TMA 2D swizzle transpose；
6. TMA double-buffer GEMM；
7. Cluster multicast；
8. Blackwell gather4/im2col/sub-byte；
9. Blackwell TMA + `tcgen05` + TMEM。

每一级保留普通 load/store baseline，避免“用了新指令所以一定更快”的自证循环。

下一篇：[Cluster、DSM 与 TMEM](08-cluster-dsm-tmem.md)。
