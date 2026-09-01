# 第六部分 CUDA 特殊指令（PTX）

> 对应 PTX ISA 手册。**这一章不要死记指令语法。** 每学一条指令，都按四个问题过一遍：它对应什么硬件电路？它是在哪一代架构因为什么问题被引入的？为什么要这样设计（而不是别的设计）？前面几部分已经把大部分背景铺垫好了，这一部分负责把它们落到具体的 PTX 指令上，作为一份"因果关系速查表"，而不是从零讲解。

## 什么是 PTX：先厘清它在软件栈中的位置

PTX（Parallel Thread Execution）是 NVIDIA 定义的一种**虚拟指令集架构（ISA）**——它不是任何具体 GPU 直接执行的机器码，而是介于 CUDA C++ 源码和真正的机器码 SASS 之间的一层中间表示（IR），类似于 LLVM IR 在 CPU 编译体系中的角色。`nvcc` 把 CUDA C++ 编译成 PTX，`ptxas`（或运行时的 JIT 编译器）再把 PTX 编译成特定架构的 SASS 机器码。PTX 的存在让 NVIDIA 可以在不破坏向前兼容性的前提下，让同一份编译产物（PTX）能被未来新架构的 `ptxas`/JIT 重新编译，充分利用新硬件特性（这是第八部分要详细展开的内容）。

## 6.1 Warp 级指令

### 6.1.1 `shfl.sync`（Shuffle）

- **对应硬件**：SM 内的 Warp 内部数据交换网络（Shuffle Network），允许 Warp 内线程之间直接交换寄存器数据，不经过 Shared Memory。
- **何时出现**：`shfl`（无 `.sync` 后缀）最早在 Kepler（SM 30）引入，是第一次给出"不经过共享内存做线程间通信"的手段；Volta 引入 Independent Thread Scheduling 后，隐式假设 Warp 内线程锁步执行不再安全，因此改为要求显式传入参与线程掩码的 `shfl.sync`。
- **为什么这样设计**：在 Shuffle 出现之前，Warp 内线程交换数据必须经过 Shared Memory（写入 + `__syncthreads()`/`__syncwarp()` + 读取），代价是至少两次访存延迟外加同步开销。Shuffle 指令直接在寄存器级别做交换，延迟只有一个周期量级，是 Warp 级归约（reduction）、扫描（scan）等算法的性能基石（第十二部分会具体分析基于 Shuffle 的 Reduce 实现）。
- **常见变体**：`shfl.sync.up/down/bfly/idx`——分别对应"从更低/更高 lane 取值"、"蝶式交换（用于树形归约）"、"任意 lane 索引取值（广播）"。

### 6.1.2 `vote.sync`

- **对应硬件**：Warp 内的投票/规约网络，把 32 个线程的一个布尔谓词聚合成一个结果。
- **变体**：`vote.sync.all`（是否所有参与线程谓词都为真）、`vote.sync.any`（是否至少一个为真）、`vote.sync.uni`（是否所有参与线程谓词一致，即 Warp 没有在这个条件上发生分歧）、`vote.sync.ballot`（返回一个 32-bit 掩码，每一位代表对应线程的谓词结果）。
- **为什么出现**：很多算法需要知道"Warp 内是否所有/任意线程满足某个条件"（比如提前退出循环、稀疏数据的压缩），如果不用 `vote`，需要写入 Shared Memory 再做归约，代价高得多。`ballot` 返回的掩码本身也是 `shfl`/`match` 等其它 Warp 级指令所需的"参与线程掩码"的常见来源。

### 6.1.3 `match.sync`

- **对应硬件**：Warp 内的比较/分组网络。
- **What**：让 Warp 内每个线程提供一个值，指令返回一个掩码，标记出"哪些线程的值和自己相同"（`match.any`）或者要求所有参与线程的值必须相同（`match.all`）。
- **典型用途**：判断 Warp 内线程访问的地址是否重复（用于自定义的访存去重/合并逻辑），或者实现无需 Shared Memory 的 Warp 内分组聚合。

## 6.2 Barrier / 同步指令

### 6.2.1 `bar.sync`

- **对应硬件**：SM 内的硬件屏障单元，是 CUDA C++ `__syncthreads()` 编译后对应的 PTX/SASS 指令，作用于 Block（CTA）范围。
- **参数**：`bar.sync barrier_id{, thread_count}`——现代架构支持多个独立的硬件屏障（不止一个），允许 Block 内的子集线程各自同步（Cooperative Groups 的 tiled/coalesced group 同步就依赖这个能力，见第七部分）。

### 6.2.2 `mbarrier`（内存屏障对象）

- **对应硬件**：Ampere 引入的、位于 Shared Memory 中的**异步事务感知屏障对象**，Hopper 进一步增强为可以感知 TMA 等异步搬运"字节数"完成情况的**Asynchronous Transaction Barrier**。
- **Why 出现**：`__syncthreads()`/`bar.sync` 只能表达"所有线程都执行到了这一行代码"，无法表达"某个异步操作（如 `cp.async`/TMA）产生的数据已经全部到达"这种更细粒度、和异步硬件深度绑定的完成状态。
- **How**：`mbarrier.init`（初始化，指定需要多少个"到达"才算完成）、`mbarrier.arrive`（线程到达，计数减一；Hopper 上还有 `mbarrier.arrive.expect_tx` 用于声明"我期望有 N 字节的异步数据即将到达，请等这些字节真正写完才算完成"）、`mbarrier.try_wait`/`mbarrier.test_wait`（非阻塞或阻塞地检查/等待屏障状态）。
- **Evolution**：这是第三部分反复提到的、贯穿 `cp.async`、TMA、WGMMA 的核心同步机制——没有 `mbarrier`，Hopper 就无法把"数据搬运完成"这种异步硬件事件，以一种线程可以廉价查询/等待的方式暴露出来。

> 这里的“屏障”同时涉及 phase、arrival 与异步 transaction，不能简化成 `bar.sync` 的替代。CTA/Warp/Cluster barrier、`fence`/proxy fence、TMA/WGMMA/`tcgen05` completion 的完整 CUDA→PTX→SASS 对照见[第十三部分：NVIDIA GPU 同步手册](./part13-synchronization-handbook/00-overview.md)。

## 6.3 Memory 指令

| 指令 | 含义 | 出现动机 |
|---|---|---|
| `ld.global` / `st.global` | Global Memory 读/写 | 基础访存指令，支持多种缓存修饰符（`.ca`/`.cg`/`.cs`/`.lu`/`.cv`）控制数据在 L1/L2 的缓存策略 |
| `ld.shared` / `st.shared` | Shared Memory 读/写 | 基础片上访存 |
| `cp.async` | 异步 Global→Shared 拷贝（第三部分已详述） | Ampere 引入，去掉寄存器中转 |
| `cp.async.bulk.tensor.*` (TMA) | 硬件 DMA 批量张量拷贝（第三部分已详述） | Hopper 引入，去掉逐线程地址生成 |
| `prefetch` / `prefetch.global.L2` | 软件预取提示，把数据提前取入指定层级 Cache 但不阻塞、不实际使用数据 | 用于隐藏已知未来会访问、但当前还不需要用的数据的延迟，是编译器/程序员对 Cache 替换策略的显式提示 |

`ld.global` 的缓存修饰符值得展开一下，因为它直接体现"程序员可以对本应透明的 Cache 做显式提示"这一 GPU 特色（对应第三部分 3.1.7）：

- `.ca`（cache at all levels）：默认行为，尝试写入 L1 和 L2。
- `.cg`（cache at global level）：绕过 L1，只写入 L2——用于预期这份数据不会被同一 SM 上其它 Warp 复用、或者不希望它把 L1 中更有价值的数据挤出去的场景。
- `.cs`（cache streaming）：提示这是"流式"数据，只用一次，标记为优先淘汰。
- `.lu`（last use）：提示这是对这个地址的最后一次访问，读取后可以立即让出 Cache 行占用的空间。

## 6.4 Tensor 指令

| 指令 | 引入架构 | 协作颗粒度 | 操作数来源 | 备注 |
|---|---|---|---|---|
| `mma.sync` | Volta 起 | Warp | 寄存器 | 原子形状从 Volta 的 `m8n8k4` 逐步扩展到 Ampere 的 `m16n8k*` |
| `ldmatrix` | Ampere | Warp | Shared Memory → 寄存器 | 专为 Tensor Core 操作数排布定制的加载指令（第五部分 5.4） |
| `wgmma.mma_async` | Hopper | Warp Group (128线程) | Shared Memory（描述符）/寄存器 | 异步，需配合 `fence`/`commit_group`/`wait_group` |
| `tcgen05.mma` / `.ld` / `.st` / `.cp` / `.alloc` | Blackwell（**仅数据中心** `sm_100a`/`sm_103a`） | 单线程可发起 | Shared Memory | 结果写入 TMEM，需要显式分配/释放 TMEM 空间；**消费级 Blackwell（`sm_120`，RTX 50 系列）不支持这组指令**，仍走 `mma.sync.aligned.block_scale` |

这张表和第五部分的演进表是同一件事的两个角度——第五部分讲"为什么变"，这里给出"具体指令名"这个查表入口。

> 本节是指令索引，不替代 operand layout、协作约束、同步协议与反汇编验证。按 WMMA/`mma.sync`/`ldmatrix` → WGMMA → `tcgen05` 的编程路线见[第五部分 Tensor Core 指令手册](./part05-tensor-core-handbook/00-overview.md)；CUDA→PTX→SASS 并非跨 Toolkit/架构的一一映射。

## 6.5 Atomics 原子指令

- **`atom`**（如 `atom.global.add.f32`、`atom.shared.cas.b32`）：对指定地址执行"读-改-写"的原子操作（加、减、最值、CAS 比较交换等），保证在多个线程/多个 SM 并发访问同一地址时不会出现数据竞争导致的错误结果。
- **`red`**（reduction）：语义上与 `atom` 类似，执行"读-改-写"，但**不需要返回操作前的旧值**——如果程序员确实不需要旧值，`red` 可以让硬件省去"把旧值传回寄存器"这一步，理论上开销更低。
- **为什么原子操作在 GPU 上特别重要**：GPU 的海量并行线程天然会频繁出现多个线程需要更新同一个共享结果（如直方图统计、全局计数器、稀疏 scatter-add）的场景，原子指令是唯一能保证正确性、同时又不需要引入昂贵的锁机制的手段。原子操作的硬件实现随架构不断优化——早期架构上竞争激烈的原子操作性能很差（因为要串行化处理同一地址的所有并发请求），后续架构逐代增强了原子操作单元的吞吐量，并在 L2 层面提供原子操作的执行能力（不需要都在 SM 一侧排队等待）。

## 6.6 Special Register 特殊寄存器

| 寄存器 | 含义 | 用途 |
|---|---|---|
| `%laneid` | 当前线程在其所属 Warp 内的编号（0-31） | Warp 内分工、Shuffle/Vote 指令的手工实现、判断是否是 Warp 内第 0 号线程（常用于"只让一个线程做某事"的模式，如 TMA 发起） |
| `%warpid` | 当前 Warp 在 SM 内的编号 | Warp 级别的资源分工（如 Warp Specialization 中区分生产者/消费者 Warp Group） |
| `%smid` | 当前线程所在的 SM 编号 | Persistent Kernel（常驻内核）判断自己被分配到了哪个 SM，从而决定要处理的数据分片；也用于负载均衡和调试 |
| `%clock64()` | 64 位时钟周期计数器 | Kernel 内部的细粒度计时（微基准测试、手工 profile），比 Host 端计时更精细，但要注意不同 SM 的时钟并不严格同步 |

## 6.7 学习方法论：不要死记，要串因果链

这一部分刻意做成一份"速查表 + 因果关系"的组合，而不是逐条指令的语法手册（语法细节请查阅官方 PTX ISA 文档，且随版本迭代变化）。真正需要内化的是这样一种思维习惯——每次遇到一条新 PTX 指令，问自己：

1. 它是哪一代架构、为了解决什么具体瓶颈引入的？（对照第三/四/五/十部分）
2. 它操作的是哪一层存储、依赖哪个硬件电路？（对照第一部分的硬件组成）
3. 它的协作颗粒度是线程、Warp 还是 Warp Group？（这个颗粒度本身往往就是它存在的意义所在）
4. 它的"上一代替代方案"是什么，新指令具体省掉了什么开销？

带着这套方法论，下一部分我们看软件工程师实际会用到的 C++ 层 API，以及它们和这一章 PTX 指令之间的对应关系。
