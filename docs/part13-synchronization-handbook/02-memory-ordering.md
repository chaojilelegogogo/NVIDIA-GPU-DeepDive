# 13.2 Memory Ordering：不等待，也能让访问有序

> Fence 不会把其它线程叫到同一行代码。它只规定：在指定 scope/proxy 中，哪些内存操作必须先被观察到。

## 13.2.1 三个不能混淆的概念

| 概念 | 解决的问题 | 是否等待其它执行者 |
|---|---|---|
| execution barrier | 谁可以继续执行 | 是 |
| fence / release / acquire | 写与读的观察顺序 | 否 |
| async wait | DMA/Tensor 工作是否完成 | 等待异步引擎或其通知 |

典型 producer/consumer 的正确结构不是“只加一个 fence”，而是：producer 写 payload → release → 发布 flag；consumer 观察 flag → acquire → 读 payload。发布/观察 flag 通常使用带内存序的 atomic；具体 C++ 形式应优先使用 CUDA 支持的原子/同步库接口，而不是假设普通 volatile 读写能建立跨线程同步。

## 13.2.2 `membar` 到 `fence`

| 字段 | `membar` | `fence` |
|---|---|---|
| 定位 | 旧 PTX 内存栅栏表示 | 现代 PTX 内存模型表示 |
| scope | 常见 `.cta` / `.gl` / `.sys` | `.cta` / `.cluster` / `.gpu` / `.sys` 等，随 PTX 版本扩展 |
| 语义 | 对指定范围内访问排序 | 结合 memory ordering qualifier 表达更精确的排序 |
| CUDA 映射 | 旧代码常见于 `__threadfence*` 的编译结果 | 没有稳定的一对一 CUDA intrinsic |
| 推荐 | 阅读旧 SASS/PTX 时理解 | 新 PTX 优先使用规范版本支持的 `fence` 形式 |

不要把 `.gl` 与现代 `.gpu` 简单当作同义词，也不要从某版 ptxas 的降级结果反推源级语义。应以代码所针对的 PTX ISA 版本为准。

## 13.2.3 在具体原语中应用 Order、Scope 与 Proxy

Order/Scope/Proxy 是三个独立问题：

```text
Order：同一线程中，哪些访问不能跨过同步操作？
Scope：哪些观察者可以依赖这个顺序？
Proxy：生产者和消费者是否通过同一种访问方法触碰内存？
```

只有三项都匹配，发布协议才可能成立。例如 device-scope release/acquire 解决两个 CTA 的普通 global-memory 发布；如果 payload 实际由 TMA async proxy 写入，还必须先等待 TMA completion，并处理规范要求的 proxy 交接。

### 13.2.3.1 Order：`relaxed/acquire/release/acq_rel/seq_cst`

先以同步操作为中点看箭头方向：

```text
release： 之前的读写 ─────→ [同步操作]
acquire： [同步操作] ─────→ 之后的读写
acq_rel： 之前的读写 ─────→ [读改写/栅栏] ─────→ 之后的读写
```

| CUDA C++ order | 约束 | 常见用途 | 不提供什么 |
|---|---|---|---|
| `memory_order_relaxed` | 只保证该 atomic 操作的原子性和该对象的 modification order | 纯计数器、索引分配 | 不发布/获取其它 payload |
| `memory_order_release` | 当前线程中在它**之前**的相关读写，不能在观察顺序上越过这次发布 | producer 写 payload 后发布 flag；解锁 | 不约束后续访问；不能单独让 consumer 同步 |
| `memory_order_acquire` | 当前线程中在它**之后**的相关读写，不能在观察顺序上提前到这次获取之前 | consumer 读到发布 flag 后读取 payload；加锁 | 没读到匹配 release 的值时，不会凭空取得 payload |
| `memory_order_acq_rel` | 同时具有 release 和 acquire；通常用于 atomic RMW | `exchange`、成功的 CAS、阶段状态转换 | 不是“先 acquire、后 release”；方向恰好相反 |
| `memory_order_seq_cst` | acquire/release 约束之外，相关 seq_cst 操作还参与一个一致的全序 | 确实依赖全局 SC 推理的算法 | 不等于全设备 execution barrier，通常也更昂贵 |

`memory_order_consume` 在 C++ API 中也可能出现，但工具链通常按 acquire 处理，且其依赖语义长期不适合作为 CUDA 教学和可移植优化基础；本手册不把它作为独立工程选择。

不同指令只接受 order 的子集：

| PTX 操作 | 常见合法 order |
|---|---|
| load | `.relaxed` / `.acquire` |
| store | `.relaxed` / `.release` |
| atomic RMW | `.relaxed` / `.acquire` / `.release` / `.acq_rel` |
| `fence` | `.acquire` / `.release` / `.acq_rel` / `.sc` |

PTX atomic 通常没有 `.seq_cst` qualifier。CUDA C++ `memory_order_seq_cst` 常被编译为 `fence.sc.<scope>` 加一个较弱的 load/store/atomic。例如本仓库 CUDA 13.0 的实验中，device-scope seq_cst store 生成为：

```ptx
fence.sc.gpu;
st.relaxed.gpu.b32 [addr], value;
```

因此应该看完整指令序列，而不是只搜索一条“seq_cst atomic”。

release/acquire 形成跨线程同步还需要：acquire 操作从匹配 release（或其 release sequence）发布的 atomic 值中读取，并且两端 scope 足够。仅仅“一边写 release、另一边执行过 acquire”并不自动建立关系。

### 13.2.3.2 Scope：`thread/cta/cluster/gpu/sys`

Scope 回答“谁能依赖这个 order”，不回答数据位于 shared 还是 global，也不让这些线程自动会合。

| CUDA C++ scope | PTX scope | 覆盖范围 | 典型场景 |
|---|---|---|---|
| `cuda::thread_scope_thread` | 没有对应的通用跨线程 PTX scope | 仅当前线程 | 主要用于泛型库接口，不建立线程间同步 |
| `cuda::thread_scope_block` | `.cta` | 同一 CTA/thread block | CTA 内 flag、shared/global atomic 协调 |
| `cuda::thread_scope_cluster`（规划中） / PTX `.cluster` | `.cluster` | 同一 thread-block cluster | Hopper+ Cluster/DSM 协作 |
| `cuda::thread_scope_device` | `.gpu` | 同一 GPU 上的线程 | 不同 CTA 间 global-memory 发布 |
| `cuda::thread_scope_system` | `.sys` | CPU、当前 GPU，以及平台支持的 peer/system 观察者 | GPU↔CPU、peer/system-scope 协调 |

CUDA 13.0 的 libcu++ 已有内部 `__thread_scope_cluster_tag`，但公开 `thread_scope` 枚举尚未包含 `thread_scope_cluster`。文档语义仍以 PTX `.cluster` 为准；配套 case06 因此对 cluster 路径使用内联 `atom.add.relaxed.cluster`，而不是不存在的 C++ scope 枚举值。

旧 PTX `membar` 使用 `.cta/.gl/.sys`；现代 `fence` 使用 `.cta/.cluster/.gpu/.sys`。在 `sm_70+` 的规范中 `membar.gl` 与 `fence.sc.gpu` 兼容，但阅读旧代码时仍应保留“旧 level 与现代 scope 属于不同语法体系”的意识。

Scope 选择规则：

```text
同一 CTA                         → .cta
不同 CTA、但同一已建立的 cluster → .cluster
同一 GPU 上任意 CTA              → .gpu
GPU 与 CPU/peer/system           → .sys
```

- scope 小于参与者范围：同步不成立；
- scope 大于参与者范围：通常语义成立，但可能付出额外成本；
- warp 不是 PTX 通用 memory scope。Warp 会合/交接应使用 `__syncwarp(mask)` 等执行或 collective 原语；
- `.cta` atomic 可以位于 global memory；scope 与 address space 是两个不同维度；
- `.sys` 只定义范围，不保证任意内存都具备 CPU/peer 原子可达性，还要满足分配类型、平台和设备能力。

### 13.2.3.3 Proxy：`generic/async/tensormap/alias/fabric`

Proxy 不是“另一种 scope”，也不是 cache level。它是 PTX memory model 给**内存访问方法**贴的抽象标签。同一地址若由不同 proxy 访问，普通 release/acquire 不一定足以把访问顺序传递过去。

| Proxy/访问类别 | 典型操作 | 要点 |
|---|---|---|
| generic proxy | 普通 `ld/st/atom/red`，大多数 CUDA C++ 指针访问 | 默认路径；同 proxy 发布通常使用普通 order + scope |
| async proxy | `cp.async.bulk`/TMA 的数据搬运、WGMMA async 等规范明确标为 async 的操作 | generic↔async 交接可能需要 `fence.proxy.async`，并且必须单独等待异步完成 |
| tensormap proxy | TMA 对 `CUtensorMap` descriptor 的读取，以及 tensormap 修改指令 | descriptor 被 generic/host/device 修改后，使用前可能需要 `fence.proxy.tensormap::generic` |
| alias proxy 情形 | 同一物理位置经不同虚拟地址 alias 访问 | 需要 `fence.proxy.alias` 建立别名路径间顺序 |
| fabric proxy | PTX 9.x fabric 操作 | 使用 fabric 方向的 proxy fence；仅适用于相应新架构和系统 |
| texture/surface 等特殊访问方法 | texture/surface 指令 | 它们也是区别于普通 generic 的访问方法，按各自规范处理一致性 |

一个很容易误解的细节：**名字里有 async，不代表一定属于 async proxy。** 当前 PTX 规范中，非 bulk Ampere `cp.async` 被定义为 weak operation，并归入 generic proxy；`cp.async.bulk`/TMA 才明确在 async proxy 中执行。必须查具体指令说明，不能根据 mnemonic 猜。

Tensor Core/TMEM 也不能被笼统称为一个统一的“Tensor proxy”。WGMMA、`tcgen05`、TMEM 各有专用 fence/commit/wait 规则；只有 PTX 明确赋予某个 proxy 的操作，才按该 proxy 的规则推理。

### 13.2.3.4 Proxy fence 与 async completion 的分工

```text
proxy fence：规定 generic 与 async 等不同访问方法之间的观察顺序
completion ：确认异步引擎是否真的完成了这次工作
barrier    ：确认所需执行参与者是否到达交接点
```

三者不能互相替代。以 TMA global→shared 为例：

```text
TMA 发起
  → mbarrier transaction completion（数据搬完）
  → 规范定义/要求的 async→generic proxy 交接
  → CTA consumer 被协议允许读取
```

部分 bulk/TMA 指令在“completion 被观察到”时隐含 async→generic proxy fence，因此消费者不一定要手写第二道同方向 fence；但 generic 写 shared 后再让 TMA/WGMMA 读取，常需要显式 `fence.proxy.async`。应查具体指令的 completion 条款，而不是无条件前后各加一道 fence。

用三个问题检查代码：

1. **Order**：producer 的 payload 写是否由 release 发布？consumer 是否在 acquire 成功后才读 payload？
2. **Scope**：两端是否都位于选定 scope 内？
3. **Proxy**：两端是否通过同一 proxy 访问？若不同，是否具有正确方向的 proxy fence，并且异步操作是否已经 completion？

### 13.2.3.5 Order × Scope 必须一起读

一条 PTX 指令里 order 与 scope 是并列 qualifier，不是两套互斥语法：

```text
st.release.gpu.b32 [flag], 1;
│  │       └── Scope：.gpu（device）
│  └────────── Order：.release
└───────────── Operation：store
```

| CUDA | 常见 PTX 形态 | 语义 |
|---|---|---|
| `store(release)` + `thread_scope_block` | `st.release.cta` | 发布给同 CTA |
| `store(release)` + `thread_scope_cluster` | `st.release.cluster` | 发布给同 cluster |
| `store(release)` + `thread_scope_device` | `st.release.gpu` | 发布给同 GPU |
| `store(release)` + `thread_scope_system` | `st.release.sys` | 发布给 system |
| `fetch_add(relaxed)` + `thread_scope_device` | `atom.add.relaxed.gpu` | 只做 device 范围原子性 |
| `fetch_add(release)` + `thread_scope_device` | `atom.add.release.gpu` | 原子性 + 发布此前访问 |
| `load(acquire)` + `thread_scope_device` | `ld.acquire.gpu` | 取得后约束后续访问 |
| `exchange(acq_rel)` + `thread_scope_device` | `atom.exch.acq_rel.gpu` | RMW 同时 release+acquire |

错误组合的典型形态：

```text
order 对、scope 太小：
  CTA0 用 release.cta 发布，CTA1 用 acquire.gpu 读取
  → CTA1 不能依赖这次发布

scope 对、order 太弱：
  producer 用 relaxed store flag，consumer 用 acquire load
  → 不构成 payload 发布关系
```

完整可运行矩阵见配套 case：01–05 比较 order，06 固定 order 扫 scope，07 观察 `fence.sc`，08 把 payload + release/acquire + scope 放在同一 CTA 协议里。

## 13.2.4 `__threadfence*` 的正确位置

| CUDA | 目标范围（概念上） | 用途 |
|---|---|---|
| `__threadfence_block()` | block | CTA 内、非 barrier 形式的发布协议 |
| `__threadfence()` | device | 同一 GPU 上 CTA 间 global-memory 通信 |
| `__threadfence_system()` | system | GPU 与 CPU/peer/系统参与者通信 |

这些函数解决的是写入可见顺序，不会让另一个 CTA 停止轮询，也不保证异步 TMA、copy engine 或 Tensor Core 已完成。`__threadfence()` 之后用普通非原子 `flag = 1` 不能单独构成完整跨 CTA 协议：consumer 仍要用匹配 acquire/atomic 观察 flag，避免普通读写 flag 的数据竞争。CTA 内规则清晰的 shared-memory handoff 优先使用 `__syncthreads()`；跨 CTA 设计优先考虑拆 kernel，只有确实需要 persistent/producer-consumer 协议时才使用带正确 scope 的 atomic + fence/order。

## 13.2.5 Proxy fence：跨访问代理交接

Hopper 之后，一块 shared memory 可被普通线程（generic proxy）、TMA bulk copy/WGMMA（async proxy）及 `tcgen05`/TMEM 专用路径访问。非 bulk `cp.async` 在当前 PTX memory model 中仍属于 generic proxy，不能仅凭名字分类。普通线程的顺序规则不会自动让另一访问方法按相同顺序观察数据。

| 原语族 | 目的 | 要点 |
|---|---|---|
| `fence.proxy.*` | 在两个 proxy 之间建立所需顺序 | 指定 proxy、scope 与 PTX ISA 支持的 qualifier |
| `fence.proxy.async` | 处理 generic ↔ async proxy 交接 | 常见于普通线程写 shared 后交给 TMA/WGMMA，或按指令规范处理反向交接 |
| `wgmma.fence` | WGMMA 所需的 operand/accumulator 依赖协议 | 不是通用 memory fence |

这不是“多加一道 fence 总会更安全”。proxy fence 必须放在生产者完成其访问、消费者被允许开始另一代理访问的正确边界，并结合 `mbarrier` 或 async wait；它既不替代 completion notification，也不替代 CTA/cluster participation barrier。

## 13.2.6 诊断清单

1. payload 和 flag 是否位于算法需要的地址空间？
2. producer 是否在发布 flag 前完成 payload 写，并使用 release 语义？
3. consumer 是否只在 acquire 成功后消费 payload？
4. scope 是否包含所有 producer/consumer？
5. 是否发生 generic/async/TMEM 等 proxy 切换？若是，是否有相应 proxy fence？
6. 这其实是否是“异步操作尚未完成”问题？若是，改看 [Async Pipelines](./03-async-pipelines.md)。

## 13.2.7 配套 CUDA → PTX → SASS 实验

[`src/part13-synchronization-handbook/02-memory-ordering/`](../../src/part13-synchronization-handbook/02-memory-ordering/README.md) 提供 8 个可运行 case：

| Case | 文档对应 | 验证什么 |
|---|---|---|
| 01 | plain vs release/acquire | 普通访问与带 order 的访问在 PTX 上的差别 |
| 02 | relaxed vs release RMW | 同一 `atom.add` 只改 order |
| 03 | acquire load | plain / relaxed / acquire load 三层对比 |
| 04 | acq_rel lock | `atom.exch.acq_rel` + release unlock |
| 05 | `__threadfence` vs release | 显式 fence 与附着在 store 上的 order |
| 06 | Scope | 同一 relaxed RMW 扫 `.cta/.cluster/.gpu/.sys` |
| 07 | seq_cst | `fence.sc` + 较弱访问的 lowering |
| 08 | Order × Scope payload | 同 CTA 内 payload 写 + release/acquire 发布协议 |

CMake 会同时生成 executable、PTX、CUBIN 与 SASS。这些 case 用于观察指令与最小正确性；跨 CTA 并发 litmus 需要 cooperative/persistent 设计，不在默认单 CTA 顺序 launch 里伪造。

下一篇：[Async Pipeline Synchronization](./03-async-pipelines.md)。
