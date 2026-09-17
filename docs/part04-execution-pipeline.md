# 第四部分 CUDA 执行流水线

> 很多教程不讲这部分，但这其实是现代 CUDA 性能优化最核心的内容之一。前面几部分讲的是"数据在哪、怎么搬"，这一部分讲的是"指令怎么被发射、怎么和别的指令重叠执行、延迟怎么被隐藏"。理解了这一部分，你才能真正看懂 Nsight Compute 里的 stall 分析（第九部分），也才能理解为什么要做软件流水线（Double/Triple Buffer）。

## 第四章 GPU Pipeline

### 4.1 Instruction Pipeline：指令怎么从取指到写回

和 CPU 一样，GPU 的每条指令也要经过取指（Fetch）、译码（Decode）、发射（Issue）、执行（Execute）、写回（Writeback）等阶段，但 GPU 的流水线是**为吞吐量而不是单指令延迟设计的**：GPU 的每一级流水线深度并不追求单指令走完全程的时间最短，而是追求"**每个周期都能有指令被发射**"这一稳态吞吐目标。也正因为如此，GPU 完全不做 CPU 那种昂贵的乱序执行、分支预测——它依靠的是"总有另一个 Warp 已经就绪"来填满流水线，而不是让单个 Warp 内部的指令乱序重排。

### 4.2 Warp Scheduler：每周期做什么决策

前面提到 Warp Scheduler 是"零开销线程切换器"，这里展开它每个周期实际做的事：

1. 扫描当前 Partition 内驻留的所有 Warp，找出哪些 Warp 的下一条指令**已经就绪**（不存在未满足的依赖，操作数已经准备好，见 4.3 节 Scoreboard）。
2. 从就绪 Warp 中，按照调度策略（常见如 **GTO, Greedy-Then-Oldest**——优先继续发射同一个 Warp 直到它 stall，再切换到最老的就绪 Warp，以及更早的 LRR, Loosely Round Robin）挑选一个（或两个，见 4.5 Dual Issue）Warp。
3. 把选中 Warp 的下一条指令发射到对应的执行单元（ALU/LD-ST/SFU/Tensor Core）。

如果**当前周期没有任何 Warp 就绪**，Warp Scheduler 只能空转（issue stall），这正是 Nsight Compute 里各种 "Stall Reason"（如 `stall_not_selected`、`stall_wait`、`stall_barrier`）想要暴露给你的信息（第九部分展开）。

### 4.3 Dependency 与 Scoreboard：硬件如何知道"能不能发射下一条指令"

GPU 不像 CPU 那样用复杂的乱序执行硬件（Reservation Station、Register Renaming）来处理指令依赖，而是采用一套更轻量的机制——**Scoreboard（记分牌）**，配合编译期（`ptxas`）静态生成的调度信息：

- 对于**固定延迟**的指令（大多数 ALU 运算，如 `FFMA`），编译器在编译期就知道确切的流水线延迟，直接在指令的控制码里编码一个**Stall Count**（等待多少个周期后，这条指令的结果才可以被下一条指令使用），硬件严格执行这个等待，不需要运行时判断。
- 对于**变延迟**的指令（访存指令 `LDG`、Tensor Core 指令等，延迟依赖于是否命中 Cache、总线拥堵情况，编译期无法预知确切周期数），编译器会给指令分配一个**Scoreboard/Barrier 编号**（现代架构通常有 6 个硬件 Barrier 寄存器）：这条指令完成时会"点亮"对应编号的 Barrier；需要用到这个结果的后续指令，会在自己的控制码里声明"必须等待这个 Barrier"，硬件在发射前检查该 Barrier 是否已经点亮，未点亮就不能发射（但可以调度别的 Warp）。

这套机制的意义：GPU 用**极低的硬件复杂度（几个 bit 的 Barrier 掩码 + 4 位的 Stall Count）**换来了"正确处理指令依赖"这个能力，把 CPU 上昂贵的动态依赖检测硬件，替换成了编译器静态分析 + 极简的运行时位掩码检查——这再次体现了"GPU 把晶体管花在吞吐量而不是单线程控制逻辑上"的哲学（呼应第一部分 1.1 节）。

### 4.4 Latency Hiding：Scoreboard 机制如何服务于延迟隐藏

当一个 Warp 因为等待某个 Scoreboard（比如在等一次 `LDG` 从 Global Memory 返回，几百周期）而无法发射下一条指令时，Warp Scheduler **不会让 SM 空转**，而是立刻切换去发射其它已经就绪的 Warp 的指令——这就是延迟隐藏的具体机制实现，也是为什么第二部分说"Occupancy 提供了更多可供 Warp Scheduler 选择的候选 Warp"：**Occupancy 越高，某一时刻处于"正在等 Scoreboard"状态的 Warp 越多，但只要有足够多其它 Warp 处于就绪状态，SM 依然能保持每周期都发射指令，从而把大量等待时间"藏"在了其它 Warp 的正常计算之中。**

### 4.5 Issue 与 Dual Issue

**Issue**指 Warp Scheduler 把一条指令真正送去执行的动作。现代 SM 通常每个 Partition 有一个 Warp Scheduler + 一个或两个 Dispatch Unit。**Dual Issue** 指一个 Warp Scheduler 在同一个周期内，向**两个不同的执行单元**（比如一个 ALU 管线 + 一个 LD/ST 管线，或两条独立的 ALU 管线）同时发射来自同一个（或两个不同）Warp 的指令，从而在不增加 Warp Scheduler 数量的前提下提升每周期的指令吞吐（IPC）。这依赖于编译器在调度时,识别出没有相互依赖、且分别对应不同执行单元的指令对,把它们安排到能被同时发射的位置。

### 4.6 Tensor Core Pipeline：一条独立的执行管线

Tensor Core 从 Volta 起作为**独立于标量 ALU 管线之外的一条专用执行单元**存在。这意味着：一个 Warp 发射一条 Tensor Core 指令（如 `mma.sync` 或 `wgmma.mma_async`）后，理论上标量 ALU 管线是空闲的，可以被同一 SM 上其它 Warp 的普通计算指令占用——这是"让 Tensor Core 计算和标量计算并发"这一优化思路（如 GEMM 中 Epilogue 的类型转换/激活函数计算与下一个 tile 的 Tensor Core 乘加重叠）的硬件基础。Hopper 的 `wgmma.mma_async` 和 Blackwell 的 `tcgen05.mma` 进一步把 Tensor Core 变成一条完全**异步**的管线——发起之后，标量/访存管线不必等待它完成就可以继续做别的事，这与 3.9/3.11 节讲的 TMA/TMEM 是同一条"计算和数据搬运彻底解耦、各自流水"设计思路在计算侧的体现。

### 4.7 Memory Pipeline 与 DMA

LD/ST Unit 是处理访存指令的专用流水线，负责地址生成、Coalescing 判断、发起对 L1/L2/Global Memory 的事务请求。从 Ampere 的 `cp.async` 到 Hopper 的 TMA，本质上是新增了**独立于 LD/ST Unit 之外的 DMA 通路**——TMA 单元可以说是一个更"重"、更专用化的 DMA 引擎，直接内嵌在 SM 中，专职处理规整的批量张量搬运，把原本要占用 LD/ST Unit 和 Warp 发射带宽的工作彻底移出主执行流水线。这条"新增独立于主流水线之外的专用管线"的模式，会在你理解了 Tensor Core 管线（4.6）和 TMA（3.9）之后变得非常清晰——它们是同一种架构演进策略（**专用异步硬件单元不断从通用执行流水线中"拆分"出去**）在计算侧和访存侧的两个体现。

### 4.8 Producer-Consumer 模式与软件流水线

有了异步搬运（`cp.async`/TMA）和异步计算（`mma.sync`/`wgmma`/`tcgen05.mma`）之后，一个自然的编程范式是把 kernel 拆成**生产者（Producer）**——负责发起下一批数据的异步搬运，和**消费者（Consumer）**——负责用当前已经就绪的数据做计算，两者通过屏障（`cp.async.wait_group` 或 `mbarrier`）协调，在时间上重叠执行：

```
时间轴 →
Producer:  [搬运 tile 0] [搬运 tile 1] [搬运 tile 2] [搬运 tile 3] ...
Consumer:               [计算 tile 0] [计算 tile 1] [计算 tile 2] ...
                         ↑ 只要 tile 0 就绪，计算就能立刻开始，不用等所有搬运完成
```

Hopper 的 Warp Group 编程模型进一步把这种 Producer/Consumer 模式**显式硬件化**——一个 CTA 内的不同 Warp Group 可以分别专职做"搬运（发起 TMA）"和"计算（发起 WGMMA）"，通过 `setmaxnreg`（第六/七部分详述）动态调节各自的寄存器配额（搬运 Warp Group 几乎不需要寄存器，可以把配额让给计算 Warp Group），这是 FlashAttention-3、CUTLASS 3.x 在 Hopper 上广泛采用的"Warp Specialization"设计范式（第十二部分会具体分析源码）。

### 4.9 Double Buffer / Triple Buffer / Ping-Pong

- **Double Buffer（双缓冲）**：准备两份 Shared Memory 空间（Buffer A、Buffer B）。计算 Buffer A 中的数据时，同时异步搬运下一批数据进 Buffer B；下一轮反过来。这样搬运时间可以被计算时间完全掩盖（前提是搬运耗时不超过计算耗时，即 kernel 是 Compute Bound 或两者接近平衡）。
- **Triple Buffer（三级缓冲）/ N 级流水线**：当单纯两级缓冲不足以掩盖足够的延迟（比如搬运延迟波动大，或希望进一步提前发起更多批次搬运以增加余量）时，扩展到 3 级或更多级缓冲，让"发起搬运"比"消费数据"提前更多步，代价是消耗更多 Shared Memory 空间——这是一个**用片上存储容量换流水线深度/延迟容忍度**的经典权衡，在 GEMM Kernel 中的 stage 数选择（2-stage/3-stage/4-stage）就是这个权衡的直接体现。
- **Ping-Pong**：广义上是双缓冲思想的一种命名方式，强调"两个缓冲区角色互换"这个动作本身（这一轮 A 是消费区、B 是生产区，下一轮互换）。在 Warp Specialization 语境下也常指"两个 Warp Group 交替扮演生产者/消费者角色"的调度模式。

### 4.10 Blackwell 的新 SM Pipeline：`tcgen05` 异步流水线与 CTA Pair

前面几节讲的流水线思想（Producer/Consumer、Double Buffer、Warp Specialization）在 Hopper 上已经相当成熟，但 Blackwell 数据中心芯片（`sm_100a`/`sm_103a`，第三部分 3.11 节已说明消费级 `sm_120` 不具备本节内容）在 SM 内部的流水线结构上又做了一次实质性调整，直接对应第三部分 3.11 节 TMEM 的引入：

**流水线阶段的变化**：Hopper 上一次 WGMMA 计算的"生命周期"是 `TMA 搬运 → Shared Memory → wgmma.mma_async 读取操作数、累加进寄存器 → 寄存器参与下一轮或写回`；Blackwell 上变成了 `TMA 搬运 → Shared Memory → tcgen05.mma 读取操作数、累加进 TMEM → tcgen05.ld 按需把结果取回寄存器 → 写回`——多出了一段**"计算单元与专属存储之间的独立数据通路"**，这段通路（TMEM 的读写）不占用寄存器堆的端口，也不占用 Shared Memory 的带宽，是一条相对于 Hopper 全新的、专属于 Tensor Core 的流水线支路。

**发起颗粒度的变化**：Hopper 的 `wgmma.mma_async` 必须由整个 Warp Group（128 线程）集体发射；数据中心 Blackwell 中，某些 `tcgen05.mma` 变体可以由**单个线程**发起（因为结果去向是 TMEM，不再固定落在发起者所在 Warp 的私有寄存器里）。这扩大了“谁负责发出计算指令”和“哪些线程后续会用到结果”的解耦空间，但**不**意味着任意线程可无条件触发任意 Tensor 操作：具体变体仍可能要求 CTA group/topology、TMEM 分配、TMA completion 和 commit/fence/wait 协议。完整边界见[第五部分 5.6](./part05-tensor-core-handbook/06-blackwell-tcgen05.md)与[第十三部分 13.4](./part13-synchronization-handbook/04-tensor-synchronization.md)。

**CTA Pair 引入的跨 SM 流水线阶段**：Blackwell 允许 TPC 内两个相邻 SM 上的 CTA 结成"CTA Pair"，共享同一份 Tensor Core 输入操作数（第三部分 3.10/3.12 节）。这意味着 SM 内部的流水线视角需要扩展为"TPC 级流水线"：一份数据被 TMA 搬进其中一个 SM 的 Shared Memory 后，通过 TPC 内专用互联直接喂给两个 SM 的 Tensor Core，而不需要各自重复搬运——这是流水线的"生产者"阶段从"单 SM 内的一个 Warp Group"进一步扩展为"一对 SM 共享的一次搬运"。

这一节想传达的核心认知：**每一代新增的"专用硬件通路"，都会相应地在软件可见的流水线阶段上多出一段**——理解了 Hopper 的 TMA+WGMMA 流水线之后，Blackwell 的变化并不是推倒重来，而是在同一套 Producer/Consumer 骨架上，把"计算结果的归宿"从寄存器换成了 TMEM，把"发起计算的颗粒度"从 Warp Group 放松为单线程，把"数据共享的范围"从单 SM 扩展到了 CTA Pair——这三个变化分别对应第三部分、第五部分反复强调的三条主线（搬运/算力/协作范围）在流水线视角下的统一体现。

### 4.11 Pipeline Programming：软件接口

CUDA 提供 `cuda::pipeline`（`<cuda/pipeline>`）这一 C++ 抽象，把"发起若干个异步拷贝 → 提交（producer_commit）→ 等待（consumer_wait）→ 释放（consumer_release）"这套协议封装成易用的接口，底层自动生成对应的 `cp.async`/`mbarrier` 指令序列，支持指定流水线级数（stage 数）。这是第七部分会详细展开的 API，这里先建立"它是在软件层面实现本章讲的 Producer/Consumer + 多级缓冲这一整套硬件机制"这个认知。

### 4.12 小结：这一部分回答了什么问题

- 为什么 GPU 不需要 CPU 那样的乱序执行硬件，也能高效地处理指令依赖？（Scoreboard + 静态 Stall Count，用极简硬件换取吞吐量）
- Occupancy 高为什么有助于隐藏延迟？（更多就绪 Warp 可供调度器选择）
- 为什么 Tensor Core、TMA 都被设计成"异步、独立于主流水线"的单元？（让搬运和计算能在时间上重叠，这是流水线思想在架构层面的落地）
- Double/Triple Buffer、Warp Specialization 这些编程技巧，本质上是把第三部分讲的"异步搬运机制"和这一部分讲的"独立流水线"两者结合起来，在软件层面显式构建计算与访存的重叠。

下一部分，我们把视角切换到**算力**这一侧——看 Tensor Core 这个专用计算单元，是怎么从 Volta 的"能做矩阵乘法"一路演进到 Blackwell 的"需要专属内存、专属指令集、专属线程组织方式"的。
