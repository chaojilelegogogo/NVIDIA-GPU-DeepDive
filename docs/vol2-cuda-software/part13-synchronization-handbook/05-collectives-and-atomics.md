# 13.5 Collective 与 Atomic：局部协同、原子性和发布协议

## 13.5.1 Warp collectives 的同步契约

| PTX | CUDA 常见入口 | 目的 | 关键契约 |
|---|---|---|---|
| `shfl.sync.*` | `__shfl_sync`、`__shfl_up/down/xor_sync` | lane 间交换寄存器值 | source lane、width 与 mask 必须符合参与协议 |
| `vote.sync.*` | `__all_sync`、`__any_sync`、`__ballot_sync` | 对谓词投票 | mask 中非退出 lanes 必须兼容参与 |
| `match.sync.*` | `__match_any_sync`、`__match_all_sync` | 按值发现等价 lanes | 值和 mask 由所有参与 lanes 提供 |
| `redux.sync.*` | 无稳定直接 intrinsic | warp 内整数规约 | 不替代数据的 general memory ordering |
| `elect.sync` | 无稳定一对一 API | 从参与 lanes 中选唯一 leader | 选出的 leader 不等于 lane 0 |
| `activemask` | `__activemask()` | 读当前活跃 mask | 快照，不是自动同步点 |

早期无 `.sync` 的 `vote` 依赖旧 warp lockstep 模型，在现代 PTX 中已弃用；使用 `vote.sync`。同理，collective 的 mask 不只是性能提示，而是参与协议的一部分：若 mask 声称某 lane 参与，但该 lane 从未执行相同 collective，结果可能未定义或死锁。

```cuda
unsigned mask = __ballot_sync(0xffffffff, active);
if (active) {
  unsigned peers = __match_any_sync(mask, key);
  // peers 表示与当前 lane key 相同且参与 mask 的 lanes
}
```

collective 能交换寄存器结果，却不会使另一个 warp/CTA 已经写好 shared/global memory。跨 warp 的 handoff 仍需 `__syncthreads()`、`cuda::barrier` 或合适的 release/acquire 协议。

`shfl.sync` 的 `.sync` 表示 mask 中 lanes 对这次 collective 的 convergence/participation 契约，不表示 general memory fence。它是 warp reduce、scan、softmax、LayerNorm 中最常见的数据交换基础：

```cuda
unsigned mask = __activemask();
for (int offset = 16; offset > 0; offset >>= 1)
  value += __shfl_down_sync(mask, value, offset);
```

若分歧后只有子集参与，应在能正确描述算法参与者的位置构造 mask；不要在每个分支内部各自调用 `__activemask()` 后假定得到同一集合。

## 13.5.2 `elect.sync` 与 warp specialization

`elect.sync` 适合“参与集合中恰好一名线程发起工作”的模式，如选择一个 leader 发起描述符操作。不要写死 lane 0，因为分支、mask 和 library composition 可能使 lane 0 不属于参与集合。leader 的工作完成后，若其它 warp/CTA 要读取其结果，仍需选择相应范围的 completion/visibility 原语。

## 13.5.3 Atomic：原子性不等于发布

| PTX | 含义 | 返回旧值 | 常见用途 |
|---|---|---|---|
| `atom.*` | 原子 read-modify-write | 是 | CAS、work queue、需要旧值的计数 |
| `red.*` | 无返回值的 reduction 原子更新 | 否 | 直方图/累加，避免返回值依赖 |
| `red.async.*` | 异步 reduction 路径（指令版本/目标相关） | 依指令定义 | 需按 PTX 指定的完成协议消费 |
| `multimem.red.*` | 多播/多内存相关归约 | 依指令定义 | 多 GPU/特殊内存路径，非通用 CTA barrier |

原子操作保证目标位置的 read-modify-write 不被并发更新撕裂，但默认并不替整个 payload 建立 producer/consumer 发布。一个常见模式是：producer 先写 payload，再以 release atomic 更新 flag；consumer acquire-load flag 成功后再读 payload。对只需要累加数值、没有额外 payload 的 `atomicAdd`，不应额外引入不必要的 system-scope fence。

### 13.5.3.1 Atomic = operation + order + scope

```text
atom.add.relaxed.cta
│    │   │       └── scope：哪些观察者可依赖
│    │   └────────── order：是否发布/获取其它访问
│    └────────────── operation：add/cas/exch/...
└─────────────────── 返回旧值的 atomic RMW
```

| 需求 | operation | order | scope |
|---|---|---|---|
| CTA 内纯计数 | add | relaxed | block/`.cta` |
| GPU 全局 work queue 索引 | add | relaxed | device/`.gpu` |
| producer 发布 payload flag | store/exchange | release | 覆盖 consumer 的最小 scope |
| consumer 获取 payload flag | load/CAS | acquire | 与 producer 匹配 |
| 锁获取/状态 RMW | exchange/CAS | acquire 或 acq_rel | 参与者范围 |
| GPU 通知 CPU | store/RMW | release | system/`.sys`，且内存/平台支持 |

`atom` 与 `red` 的 order 集合并不完全相同：需要旧值的 RMW 可表达 acquire/release/acq_rel；无返回 reduction 通常只需要 relaxed/release。实际合法组合以目标 PTX 指令 grammar 为准。

## 13.5.4 `red.async` 的边界

不要把名字中含 `async` 的任意 reduction 当成 `cp.async` 同一种完成模型。其可用架构、地址空间、是否需要显式 wait，以及可见性要求由 PTX 指令版本规定。文档或代码中应同时记录：

1. PTX ISA / CUDA Toolkit 版本；
2. 目标 `sm`；
3. 数据写入的地址空间和消费者 proxy；
4. 完成通知与 wait/fence；
5. 是否可在目标 GPU 上通过 `nvdisasm` 验证。

## 13.5.5 选择表

| 需求 | 优先原语 | 不应误用为 |
|---|---|---|
| 同 warp 的布尔投票 | `vote.sync` | CTA barrier |
| 按 key 分组 | `match.sync` | 全局 hash/atomic 的替代 |
| 从参与 lanes 选 leader | `elect.sync` | 固定 lane 0 |
| 争用计数器 | `atom` / 无返回值时 `red` | payload 发布协议 |
| 跨 CTA 发布数据 | atomic + 合适 scope 的 release/acquire | 仅 atomicAdd 或 `volatile` |

## 13.5.6 配套案例

- [`case01_warp_collectives.cu`](../../../src/part13-synchronization-handbook/05-collectives-and-atomics/case01_warp_collectives.cu)：同一 warp 中观察 activemask、shuffle、vote、match。
- [`case02_atom_vs_red.cu`](../../../src/part13-synchronization-handbook/05-collectives-and-atomics/case02_atom_vs_red.cu)：比较“使用旧值”的 `atom` 与“不使用旧值”时可能生成的 `red`。
- atomic order/scope 的完整矩阵复用 [13.2 的 8 个 case](../../../src/part13-synchronization-handbook/02-memory-ordering/README.md)，避免把同一概念维护两套不一致代码。

下一篇：[CUDA → PTX → SASS Mapping](06-cuda-ptx-sass-mapping.md)。
