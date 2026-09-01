# Part 13 / 13.2 Memory Ordering 测试

目标：并排观察 **CUDA C++ order/scope → PTX qualifier/fence → SASS atomic/MEMBAR**，同时运行最小正确性检查。

## 目录

| Case | 源码 | 比较内容 |
|---|---|---|
| 01 | `case01_plain_vs_release_acquire.cu` | 普通 load/store 与 release/acquire |
| 02 | `case02_relaxed_vs_release_rmw.cu` | relaxed 与 release `fetch_add` |
| 03 | `case03_acquire_load.cu` | plain、relaxed atomic、acquire load |
| 04 | `case04_acq_rel_lock.cu` | acq_rel exchange + release unlock |
| 05 | `case05_threadfence_vs_release.cu` | 显式 `__threadfence()` 与 release store |
| 06 | `case06_scopes.cu` | block/device/system + PTX `.cluster`（libcu++ 尚未导出 `thread_scope_cluster`） |
| 07 | `case07_seq_cst.cu` | relaxed/release/seq_cst store 与 seq_cst load |
| 08 | `case08_order_scope_payload.cu` | CTA 内 payload + release/acquire，验证 Order×Scope |

## 构建与运行

在仓库根目录执行：

```bash
cmake -S . -B build -DNVGPU_CUDA_ARCH=native
cmake --build build -j
ctest --test-dir build --output-on-failure
```

没有本机 GPU、交叉编译或希望固定结果时显式指定目标：

```bash
cmake -S . -B build-sm87 -DNVGPU_CUDA_ARCH=sm_87
cmake --build build-sm87 -j
```

不要在同一个 build 目录中反复切换 architecture；重新建目录更容易保证 PTX/SASS 可比。

## 生成 PTX 与 SASS

```bash
cmake --build build --target part13_memory_order_inspect -j
```

产物位于：

```text
build/src/part13-synchronization-handbook/02-memory-ordering/artifacts/
├── part13_case*.ptx
├── part13_case*.cubin
└── part13_case*.sass
```

快速定位：

```bash
rg 'st\.|ld\.|atom\.|fence|membar' build/**/artifacts/*.ptx
rg 'ATOM|RED|MEMBAR|FENCE|LDG|STG' build/**/artifacts/*.sass
```

## 预期观察

以下是语义形态，不保证当前 Toolkit 使用完全相同的拼写或 lowering：

| CUDA C++ | 常见 PTX 线索 |
|---|---|
| 普通 store/load | `st.global` / `ld.global` |
| relaxed device atomic RMW | `atom.relaxed.gpu.global.*`，旧目标也可能显示更弱/旧式写法 |
| release device store/RMW | `st.release.gpu.global.*` 或 `atom.release.gpu.global.*` |
| acquire device load | `ld.acquire.gpu.global.*` |
| acq_rel exchange | `atom.acq_rel.gpu.global.exch.*` |
| block/cluster/device/system scope | `.cta` / `.cluster` / `.gpu` / `.sys` |
| `__threadfence()` | PTX `membar.gl` 或现代 `fence.*.gpu` lowering |
| seq_cst | 常见为 `fence.sc.<scope>` 配合 relaxed/ordered access，具体看目标 |

SASS 不承诺“一条 CUDA order 对应一条独立 `MEMBAR`”。可能出现：

- ordering 编码在 atomic/load/store 机器指令变体中；
- ptxas 用一条或多条 `MEMBAR`/特殊控制序列实现；
- 在目标架构允许时合并、加强或重排 fence；
- 返回值未使用的 atomic RMW 被降为 reduction-only 指令。

Case 02 把 `fetch_add` 的旧值写到输出，正是为了避免最后一种情况，让 `atom` 对比更稳定。

### 本仓库当前环境的实测

使用 CUDA 13.0、`-arch=native`（本机生成 `sm_110`）验证得到：

```text
plain store       → st.global.u32
release store     → st.release.gpu.b32
relaxed RMW       → atom.add.relaxed.gpu.s32
release RMW       → atom.add.release.gpu.s32
acquire load      → ld.acquire.gpu.b32
acq_rel exchange  → atom.exch.acq_rel.gpu.b32
__threadfence()   → membar.gl
seq_cst store     → fence.sc.gpu + st.relaxed.gpu.b32
seq_cst load      → fence.sc.gpu + ld.acquire.gpu.b32
```

对应 SASS 中确实观察到 `MEMBAR.ALL.GPU`、`MEMBAR.SC.GPU`、`ATOM.E.*.STRONG.{SM/GPU/SYS}`、`LD.E.STRONG.GPU` 和 `ST.E.STRONG.GPU`。这组结果用于理解当前工具链，换 architecture 或 Toolkit 后应重新生成，不应背成固定映射。

## 每个 Case 应该理解什么

### Case 01：普通访问与发布/获取

普通 load/store 不建立跨线程同步。release store 发布此前访问；读取同一个 atomic 值的 acquire load 成功取得发布后，后续访问才能依赖该顺序。

### Case 02：relaxed 与 release RMW

两者都保证该 counter 的 atomicity。relaxed 不发布其它 payload；release 还约束此前访问不能在观察顺序上越过这次 RMW。

### Case 03：acquire load

acquire 约束**它之后**的 load/store，防止这些消费访问在观察顺序上提前到 acquire 之前。单独 acquire、但没有读取到匹配 release 发布的值，并不会凭空同步其它数据。

### Case 04：acq_rel exchange

正确方向是：

```text
exchange 之前的访问 ── release 部分 ──→ exchange
exchange ── acquire 部分 ──→ exchange 之后的访问
```

它不是“atomic 之前 acquire、之后 release”。锁获取常使用 acquire 或 acq_rel；锁释放通常只需 release store。

### Case 05：`__threadfence()` 与 release store

```text
payload store → __threadfence() → ordinary flag store
payload store → release atomic flag store
```

第一种方便观察显式 fence，但 ordinary flag 在真正跨线程程序中仍会产生数据竞争，不能独立构成完整协议。consumer 必须用合法 atomic/acquire 观察机制；实际代码优先第二种发布方式。

### Case 06：Scope

scope 必须覆盖所有依赖该顺序的参与者：

- block → `.cta`；
- cluster → `.cluster`（Hopper+）；
- device → `.gpu`；
- system → `.sys`。

scope 越大不代表越正确；过大可能增加成本，过小则同步不成立。

### Case 07：Sequential Consistency

`seq_cst` 除 acquire/release 约束外，还要求相关 seq_cst 操作参与一个一致的全序。编译器可能把它拆成 `fence.sc` 与较弱访问，而不是生成名为 `atom.seq_cst` 的单条指令。

### Case 08：Order × Scope payload 协议

同 CTA 中 producer 写 payload，再用 `thread_scope_block` 的 release store 发布 flag；consumer 用 acquire load 看到 flag 后再读 payload。观察：

```text
st.global            # payload
st.release.cta       # flag
ld.acquire.cta       # flag
ld.global            # payload
```

这是文档“Order + Scope 必须一起读”的最小可运行版本。跨 CTA 时要把 scope 升到 `.gpu/.cluster`，并处理驻留约束。

## 两个实验边界

1. 这些 executable 主要验证 API 能正常执行；kernel 依次 launch，不能用运行结果证明复杂跨 CTA memory model。
2. 真正 producer/consumer 实验不能让两个普通 CTA 无限自旋并假定同时驻留；应使用 cooperative launch、persistent-kernel 设计，或拆成两个 kernel。

配套文档：[13.2 Memory Ordering](../../../docs/part13-synchronization-handbook/02-memory-ordering.md)。
