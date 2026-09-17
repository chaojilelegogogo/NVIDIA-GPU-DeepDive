# 实战路线图：Ampere → Hopper → Blackwell 循序渐进学习

> 本文是全书的"第二种读法"。第三/四/五/六/九/十部分是按**主题**（Memory、流水线、Tensor Core、PTX、SASS、硬件史）组织的深度内容；这份路线图是按**架构代际**重新串联同一批知识点，服务于一个具体目标——**让你按 Ampere → Hopper → Blackwell 的顺序动手学习，每学完一代，先搞清楚"这一代做了什么、为什么这样做"，再问"它留下了什么问题、逼着下一代必须继续改"，从而真正建立起因果链条，而不是孤立地记住三份特性列表。**
>
> 三条 Track 分别对应三代架构，每条 Track 内部固定结构：**问题（承接上一代）→ 新特性学习清单 → 深度内容跳转 → 动手验证 → 遗留问题（过渡到下一代）**。

## 起点：Track 0 —— 在学 Ampere 之前，先确认这些基线知识

在正式进入 Ampere 之前，确保你已经清楚以下"前 Ampere 基线"（对应 Volta/Turing，详见第五部分 5.3/5.4），因为 Ampere 的每一个新特性都是相对这个基线做的改进：

- Tensor Core 从 Volta（2017）就存在了，最早的编程接口是 Warp 级的 `wmma`/`mma.sync`（原子形状 `m8n8k4`），只支持 FP16 输入（第五部分 5.3）。
- 数据从 Global Memory 搬到 Shared Memory，在 Ampere 之前只有一种方式：`ld.global` 读入寄存器，再 `st.shared` 写出——**必须经过发起线程的寄存器**（第三部分 3.7）。
- 没有软件可显式控制的"流水线"抽象，双缓冲需要程序员完全手写同步逻辑。

如果这三点还不熟悉，建议先回读第五部分 5.3~5.4 和第三部分 3.7。

---

## Track 1：Ampere（A100，`sm_80`）—— "喂饱 Tensor Core"问题第一次被正面命名

### 1.0 这一代要解决的核心问题

Volta/Turing 已经证明 Tensor Core 这条路线可行，Ampere 的任务是**大幅提高 Tensor Core 的算力密度**（第三代 Tensor Core），但很快会发现：算力提上去之后，"怎么把数据喂给它"反而成了新瓶颈——这正是本 Track 三个学习点的共同主线。

### 1.1 学习 Tensor Core（Ampere 第三代）

**学什么**：TF32/BF16 两种新数据格式、`mma.sync` 指令扩展到 Warp 级全员协作（原子形状 `m16n8k8`/`m16n8k16`/`m16n8k32`）、`ldmatrix` 专用加载指令、结构化 2:4 稀疏加速。

**深度内容跳转**：第五部分 5.4 节（TF32/BF16 的设计动机、`mma.sync` 具体形状、`ldmatrix` 为什么出现）；PTX 语法见第六部分 6.4 节表格。

**为什么排在第一个学**：因为 Ampere 后面两个新特性（`cp.async`、Async Pipeline）都是"为了喂饱这里学到的 Tensor Core"而存在的，必须先建立"Tensor Core 有多饿"这个直觉，才能理解后面两个特性的动机。

**动手验证**：
```bash
# 写一个用 wmma 或内联 mma.sync 的 tiled GEMM，编译并查看 SASS
nvcc -arch=sm_80 -cubin gemm_mma.cu -o gemm_mma.cubin
cuobjdump -sass gemm_mma.cubin | grep -E "HMMA|LDSM"
```
确认能在 SASS 里看到 `HMMA.16816`（Ampere 上 `m16n8k16` 对应的 SASS 助记符）和 `LDSM`（`ldmatrix` 对应的 SASS 指令）。用 Nsight Compute 观察这个 kernel 的 Tensor Core 利用率和访存 Stall 占比——大概率会发现 Shared Memory 访问（`LDS`/`STS`）和地址计算指令占了不小比例，这就是下一步要解决的问题。

### 1.2 学习 `cp.async`

**学什么**：`cp.async.ca`/`cp.async.cg` 两种缓存策略、`commit_group`/`wait_group` 协议、`cuda::memcpy_async` C++ 封装。

**深度内容跳转**：第三部分 3.7（问题铺垫）→ 3.8（Why/What/How/Evolution 完整展开）；C++ API 见第七部分 7.5；完整的 group commit/wait 语义见[第十三部分 13.3](./part13-synchronization-handbook/03-async-pipelines.md)。

**承接 1.1 的因果关系**：1.1 节动手验证时观察到的"访存指令占用发射带宽、寄存器被中间数据占用"问题，`cp.async` 正面回应——把 Global→Shared 的搬运改成硬件异步执行，不再路过寄存器。

**动手验证**：把 1.1 节的 GEMM 改造成用 `cp.async` 搬运 tile（而不是 `ld.global`+`st.shared`），用 `cuobjdump -sass` 确认生成了 `LDGSTS`（`cp.async` 对应的 SASS 助记符），并用 Nsight Compute 对比改造前后的寄存器使用量（`--ptxas-options=-v` 也能看到编译期报告的寄存器数变化）。

### 1.3 学习 Async Pipeline

**学什么**：Producer/Consumer 编程范式、Double/Triple Buffer、`cuda::pipeline` 的 `producer_acquire/commit`、`consumer_wait/release` 协议。

**深度内容跳转**：第四部分 4.8（Producer-Consumer 模式）、4.9（Double/Triple Buffer/Ping-Pong）、4.11（`cuda::pipeline` 软件接口）；完整代码骨架见第七部分 7.5。

**承接 1.2 的因果关系**：单独发起一次 `cp.async` 只是"这一次搬运变成异步的"，真正的收益要靠"提前发起下一批搬运、让它和当前计算重叠"才能兑现——这就是为什么 `cp.async` 之后紧跟着要学流水线编程，两者是配套的，不学流水线，`cp.async` 的异步能力基本被浪费。

**动手验证**：把 1.2 节的 kernel 扩展成 2-stage 流水线（用 `cuda::pipeline` 或手写 `commit_group`/`wait_group`），用 Nsight Systems 观察时间线上"搬运"和"计算"两个阶段是否真的重叠；用 Nsight Compute 对比流水线化前后的总耗时和 `stall_long_scoreboard` 占比。

### Ampere Track 小结：遗留了什么问题，逼着 Hopper 必须继续改

按第三部分 3.8 节 Evolution 小节的原话："`cp.async` 虽然去掉了数据路过寄存器这一步，但仍然要求**每个参与的线程各自计算自己那一份数据的源/目标地址、各自发起一条指令**"。具体拆解成三个遗留问题：

1. **搬运还是"Warp 参与"的**：32 个线程各自执行 `cp.async` 指令、各自算地址，虽然异步不阻塞，但仍占用 Warp 的指令发射带宽。
2. **协作颗粒度停留在 Warp（32 线程）**：`mma.sync` 一次只能由一个 Warp 发起，处理的 tile 大小受限，难以匹配下一代大幅提升的 Tensor Core 峰值算力。
3. **相邻 SM 之间无法共享数据**：如果两个 Block（运行在不同 SM 上）需要的数据有重叠，只能各自独立发起搬运，无法利用"物理位置很近"这个优势。

这三个问题，分别对应 Hopper 的 TMA、Warp Group/WGMMA、Cluster/DSM 三个新特性——注意它们不是孤立设计的，而是精确针对上面三条遗留问题各自提出的解法。

---

## Track 2：Hopper（H100，`sm_90a`）—— 从"Warp 参与"到"专用硬件全权负责"

### 2.0 这一代要解决的核心问题

延续上面的遗留问题：Hopper 的核心思路是把"原本由 Warp/线程亲自参与"的工作，逐步转移给专用硬件电路或更大的协作颗粒度去完成。四个新特性要按下面的顺序学，因为它们之间有严格的依赖关系。

### 2.1 学习 Warp Group（先学这个，因为后面两个特性都依赖这个新协作颗粒度）

**学什么**：Warp Group = 4 个连续 Warp（128 线程）作为一个整体去发起 Tensor Core 指令；`setmaxnreg.inc/dec` 动态寄存器配额调整；Warp Specialization（生产者 Warp Group 专职搬运、消费者 Warp Group 专职计算）的编程范式。

**深度内容跳转**：第五部分 5.5 节；第四部分 4.8 节 Producer-Consumer 在 Warp Group 层面的硬件化。

**为什么先学这个**：WGMMA 和后面的 TMA 都是围绕"以 Warp Group 为单位协作"这个新的组织方式设计的，不先建立这个概念，直接看 WGMMA 语法会觉得莫名其妙——为什么突然要求 128 个线程步调一致，而不是 32 个。

### 2.2 学习 WGMMA

**学什么**：`wgmma.mma_async` 指令、操作数直接来自 Shared Memory 的矩阵描述符（Matrix Descriptor）、`wgmma.fence`/`commit_group`/`wait_group` 协议、相比 `mma.sync` 能处理的 tile 大幅增大。

**深度内容跳转**：第五部分 5.5 节；PTX 语法与 `mma.sync` 对照见第六部分 6.4 表格；WGMMA fence/commit/wait 依赖协议见[第十三部分 13.4](./part13-synchronization-handbook/04-tensor-synchronization.md)。

**承接 2.1 的因果关系**：正因为协作颗粒度扩大到了 Warp Group，WGMMA 才能：(a) 处理比 `mma.sync` 大得多的 tile（如 `m64n128k16`），(b) 操作数直接从 Shared Memory 读取而不必先 `ldmatrix` 到寄存器——这两点都是"更大的协作颗粒度"带来的直接收益，不是无关的独立改进。

**动手验证**：对照 1.1 节的 Ampere `mma.sync` GEMM，改写一份 Hopper 版本用 `wgmma.mma_async`，编译后在 SASS 里搜索 `HGMMA`/`QGMMA`（视精度而定）或 CUTLASS 里对应的 warpgroup mma 指令，对比同样矩阵规模下两者的 Tensor Core 利用率。

### 2.3 学习 TMA

**学什么**：`CUtensorMap` 描述符构建（`cuTensorMapEncodeTiled`）、单线程发起搬运、硬件自动完成地址生成/边界检查/Shared Memory Swizzle、`mbarrier` 异步事务屏障、Multicast。

**深度内容跳转**：第三部分 3.9（完整 Why/What/How/Evolution）；PTX 指令见第六部分 6.2.2（`mbarrier`）与 6.3（`cp.async.bulk.tensor.*`）；phase/transaction completion 状态机见[第十三部分 13.3](./part13-synchronization-handbook/03-async-pipelines.md)。

**承接关系（回应 Ampere 遗留问题 1）**：TMA 直接解决 Ampere Track 遗留问题 1——"搬运还是 Warp 参与的"。现在只需要 Warp Group 中的一个线程发起一次 TMA，其余 127 个线程完全空闲。**建议把 2.2 节的 WGMMA kernel 和这一节的 TMA 结合起来**，构成"TMA 搬运 tile 到 Shared Memory → WGMMA 直接从 Shared Memory 消费"的标准 Hopper 组合，这也是 FlashAttention-3、CUTLASS 3.x 的标准范式（第十二部分 12.8/12.9 会具体分析）。

**动手验证**：构建一个 `CUtensorMap`，用 `cp.async.bulk.tensor.2d.shared::cta.global` 发起加载，配合 `mbarrier.arrive.expect_tx`/`try_wait` 等待完成，用 Nsight Compute 观察 Warp 的指令发射数是否显著下降（因为只有 1 个线程真正发指令）。

### 2.4 学习 Distributed Shared Memory（Cluster/DSM）

**学什么**：Thread Block Cluster 的声明方式（`__cluster_dims__`/`cudaLaunchAttributeClusterDimension`）、`cluster_group` API（`map_shared_rank`/`cluster.sync()`）、PTX 新地址空间 `.shared::cluster` 与 `mapa` 指令、TMA Multicast 如何依赖 DSM。

**深度内容跳转**：第三部分 3.10（完整 Why/What/How/Evolution，新增章节）；第二部分 2.2.1 节回顾"Block 不能跨 SM"这条基本规则为什么在 DSM 之下依然成立。

**承接关系（回应 Ampere 遗留问题 3）**：DSM 直接解决 Ampere Track 遗留问题 3——相邻 SM 之间此前无法共享数据，现在 Cluster 内的 Block 可以直接互相访问对方的 Shared Memory，TMA 的 Multicast 能力正是建立在这个新地址空间之上。

**动手验证**：写一个 2 个 Block 组成 Cluster 的小实验，一个 Block 往自己的 Shared Memory 写入数据，用 `cluster.map_shared_rank` 让另一个 Block 读取这份数据，验证跨 SM 直接访问确实生效（而不需要经过 Global Memory 中转）。

### Hopper Track 小结：遗留了什么问题，逼着 Blackwell 必须继续改

按第三部分 3.9 节 Evolution 小节：TMA 和 WGMMA 已经把"搬运"和"更大协作颗粒度"这两个问题解决得很好，但两个新问题浮现：

1. **累加器（Accumulator）持续膨胀**：WGMMA 处理的 tile 越来越大，累加结果占用的寄存器数量水涨船高，256KB/SM 的寄存器堆（这个容量在 Volta 到 Hopper 之间从未增长过）逐渐吃紧。
2. **精度下探到极限，需要硬件级缩放**：FP8 已经是 Hopper 能提供的最低精度，继续往 FP4/FP6 走，朴素量化误差会大到不可接受，需要硬件配合更精细的缩放机制。

这两个问题正是 Blackwell TMEM 和 NVFP4 分别要解决的。

---

## Track 3：Blackwell（数据中心 B100/B200/B300，`sm_100a`；消费级 RTX 50 系列，`sm_120`）—— 累加器搬出寄存器 + 极致低精度

> **在开始这个 Track 之前必须先明确一件事**：NVIDIA 用同一个"Blackwell"品牌名，卖两种设计差异很大的芯片。**下面 3.1（TMEM）和 3.3（新 SM Pipeline）只存在于数据中心 Blackwell（B100/B200/B300）**；如果你手上是 **RTX 5070/5080/5090（消费级 Blackwell，`sm_120`）**，这两节的内容在你的显卡上跑不起来（编译时用 `-arch=sm_120`，代码里不能出现 `tcgen05.*`），你能学到的是 3.2（第五代 Tensor Core 的 FP4/FP6 支持，消费级用 `mma.sync.aligned.block_scale` 实现同样的低精度收益）和 3.4 中不依赖 TMEM 的那部分指令。详见第三部分 3.11 节和第五部分 5.6 节的完整说明。

### 3.0 这一代要解决的核心问题

延续 Hopper Track 遗留的两个问题：寄存器堆装不下越来越大的累加器、精度需要压到 4/6 bit 但必须保证可用。

### 3.1 学习 TMEM（数据中心 Blackwell 专属）

**学什么**：TMEM 的物理组织（128 Lane × 最多 512 列，约 256KB/SM）、显式分配/释放（`tcgen05.alloc`/`dealloc`，编程模型更接近 Shared Memory 而不是寄存器）、`tcgen05.mma` 结果直接写入 TMEM、`tcgen05.ld`/`st`/`cp` 数据搬运。

**深度内容跳转**：第三部分 3.11（完整 Why/What/How/Evolution）；第五部分 5.6 节。

**承接关系（回应 Hopper 遗留问题 1）**：TMEM 直接解决"累加器持续膨胀、寄存器堆装不下"这个问题——把累加器从寄存器堆搬到一块专属的新存储里，寄存器堆不再需要为累加器预留空间。

**发起颗粒度的变化（承接 2.1 的 Warp Group 概念）**：注意这里发生了一次"反直觉"的变化——协作颗粒度不是继续扩大（Warp Group → 更大的组），而是**退回到单线程即可发起**（`tcgen05.mma`）。原因在于 Warp Group 集体发射的必要性，本质上是因为 WGMMA 的结果要落到发起者所在 Warp(Group) 的私有寄存器里，需要整组线程步调一致；而 Blackwell 的结果统一落地到所有线程都能访问的 TMEM，不再需要"发起者和结果拥有者必须是同一批线程"，于是这个约束被解除了。

### 3.2 学习第五代 Tensor Core（FP4/FP6，两种 Blackwell 都适用，但实现路径不同）

**学什么**：NVFP4 两级缩放格式（16 个 FP4 值共享一个 FP8 微块缩放因子 + 张量级 FP32 缩放）、第二代 Transformer Engine 的动态精度管理、数据中心版走 `tcgen05.mma` 消费 NVFP4，消费级版走 `mma.sync.aligned.block_scale`。

**深度内容跳转**：第五部分 5.6 节（消费级/数据中心分叉与 block-scale MMA）。

**承接关系（回应 Hopper 遗留问题 2）**：这一步直接回应"精度下探需要硬件级缩放"的问题——NVFP4 的两级缩放机制，是 Ampere TF32、Hopper FP8 这条"精度持续下探"主线的延续，只是这次伴随的量化误差已经大到必须由硬件专门设计缩放电路来兜底，不能再像 TF32 那样"几乎透明"。

**动手验证（消费级 RTX 50 也能做）**：用 `mma.sync.aligned.block_scale`（或对应 CUTLASS/CUTE 封装）写一个 FP4 GEMM，对比同规模 FP8 GEMM 的吞吐和精度损失，并记录所用格式、scale 策略和矩阵规模。

### 3.3 学习新的 SM Pipeline（数据中心 Blackwell 专属）

**学什么**：`tcgen05` 流水线的完整数据流（TMA 搬运 → Shared Memory → `tcgen05.mma` 累加进 TMEM → `tcgen05.ld` 取回寄存器做 Epilogue）、CTA Pair 引入的"TPC 级流水线"（相邻两个 SM 共享一次 TMA 搬运的结果，直接喂给两份 Tensor Core）。

**深度内容跳转**：第四部分 4.10（新增章节，完整展开这条新流水线的三个变化：存储通路、发起颗粒度、跨 SM 协作范围）；第三部分 3.10/3.12（DSM 与 CTA Pair 的关系）；第五部分 5.6。

**承接关系**：这一步把 3.1（TMEM）和 Hopper 学过的 Cluster/DSM（2.4 节）结合起来——CTA Pair 可以看作"DSM 思想在 Tensor Core 操作数共享层面的进一步下沉"：DSM 让 Block 共享 Shared Memory 中的数据，CTA Pair 让相邻 SM 直接共享 Tensor Core 的输入操作数，链路更短、专用互联延迟更低。

### 3.4 学习新的 PTX/MMA 指令

**学什么**：`tcgen05` 完整指令族——`alloc`/`dealloc`/`relinquish_alloc_permit`（TMEM 生命周期管理）、`ld`/`st`/`cp`（数据搬运）、`mma`/`mma.ws`（矩阵乘加，`.ws` 变体支持 Warp Specialization 场景）、`commit`/`fence`/`wait`（同步）。消费级 Blackwell 对应学习 `mma.sync.aligned.block_scale` 这一支持微块缩放的 `mma.sync` 扩展变体。

**深度内容跳转**：第六部分 6.4 节表格（已标注数据中心/消费级差异）；[第十三部分 13.4](./part13-synchronization-handbook/04-tensor-synchronization.md) 给出 `tcgen05` commit/fence/wait 与 TMEM 生命周期的同步边界；官方 PTX ISA 手册中 `tcgen05` 章节（随 CUDA Toolkit 版本更新，务必核对你使用的 CUDA 版本对应的手册）。

**动手验证**：在数据中心 Blackwell（或云实例）上，写一个完整的 `tcgen05.alloc → tcgen05.mma → tcgen05.ld → tcgen05.dealloc` 序列，反汇编确认 SASS 中出现的 TMEM 相关操作码（第五部分调研资料中提到的 opcode 范围，如 122-139 号段）；在消费级 RTX 50 上，对照验证同样的 GEMM 用 `mma.sync.aligned.block_scale` 实现，比较两条路径在各自硬件上的吞吐表现。

### Blackwell Track 小结：面向未来的预判

到这里，三条 Track 的因果链完整闭合。用一句话概括这条主线在 Blackwell 收尾时的状态：**算力（第五代 Tensor Core+FP4/FP6）继续指数级增长，搬运（TMA+DSM+CTA Pair）已经把"数据从哪里来"这件事基本外包给了专用硬件，这一代新解决的是"计算结果放在哪里"这个此前被忽视的环节（TMEM）**。按照这条规律，可以合理预判未来架构（Rubin 等）大概率会继续在"结果如何更高效地流转到下一步计算或写回"这个方向上做文章，也可能出现更大范围的跨 SM/跨 GPC Tensor Core 协作（CTA Pair 目前局限在同一 TPC 内的两个 SM，下一步会不会扩展到更大范围？），以及精度维度是否还有进一步下探的空间（或者转向"变精度/自适应精度"这类更精细的方案）。带着这份路线图建立的因果思维习惯，去看未来的架构发布会，你会发现自己已经具备了提前问出正确问题的能力。

---

## 三条 Track 一图流

```
Ampere (A100)                     Hopper (H100)                      Blackwell (数据中心 B200 / 消费级 RTX 50)
────────────────                  ────────────────                   ────────────────────────────────────
Tensor Core 3rd gen                                                  第五代 Tensor Core
(mma.sync, TF32/BF16, ldmatrix)                                      (FP4/FP6, NVFP4, 两种硬件路径分叉)
        │ 喂不饱                                                              ▲ 精度下探到极限需要硬件缩放
        ▼                                                                     │
cp.async                    ──问题:仍是Warp参与搬运──▶  TMA                    │
(异步搬运,去掉寄存器中转)                              (单线程发起,硬件DMA全权负责)  │
        │ 异步能力需要流水线才能兑现                            │ 累加器随WGMMA tile增大 │
        ▼                                              ▼ 而膨胀,寄存器堆装不下   │
Async Pipeline               ──问题:协作颗粒度只有Warp──▶ Warp Group + WGMMA ────┤
(Producer/Consumer,                                    (128线程协作,更大tile,   │
 Double/Triple Buffer)                                  操作数直接读SMEM)        ▼
        │ 相邻SM无法共享数据                                     │            TMEM(仅数据中心)
        │                                                       ▼            (累加器搬出寄存器堆,
        └───────────────────────────────────▶  Distributed Shared Memory      TMEM专属存储)
                                                (Cluster,跨SM共享Shared Memory,       │
                                                 支撑TMA Multicast)                   ▼
                                                                              新SM Pipeline + CTA Pair
                                                                              (tcgen05流水线,双SM协作MMA)
```

## 每个学习点对应的深度内容速查表

| 架构 | 特性 | 深度内容位置 | 消费级硬件是否可实践 |
|---|---|---|---|
| Ampere | Tensor Core（TF32/BF16/`mma.sync`/`ldmatrix`） | Part 5 §5.3 | 是（RTX 30/40/50 均可） |
| Ampere | `cp.async` | Part 3 §3.7~3.8 | 是（`sm_80`+ 均可） |
| Ampere | Async Pipeline | Part 4 §4.8~4.9、§4.11；Part 7 §7.5 | 是 |
| Hopper | Warp Group / Warp Specialization | Part 5 §5.4.1、§5.4.3；Part 4 §4.8 | 否（需要 H100/H200 或云实例） |
| Hopper | WGMMA | Part 5 §5.4.2 | 否 |
| Hopper | TMA | Part 3 §3.9 | 否 |
| Hopper | Distributed Shared Memory / Cluster | Part 3 §3.10 | 否 |
| Blackwell | TMEM | Part 3 §3.11 | **否**（仅 B100/B200/B300） |
| Blackwell | 第五代 Tensor Core（FP4/FP6/NVFP4） | Part 5 §5.5.2~5.5.3、§5.5.6 | **部分可以**（RTX 50 用 `mma.sync.aligned.block_scale` 路径） |
| Blackwell | 新 SM Pipeline（`tcgen05` 流水线/CTA Pair） | Part 4 §4.10；Part 5 §5.5.5 | **否**（仅数据中心） |
| Blackwell | 新 PTX/MMA 指令（`tcgen05` 族） | Part 6 §6.4 | **否**（RTX 50 对应学 `mma.sync.aligned.block_scale`） |

如果你手上只有消费级显卡（例如 RTX 5070），实操上可以完整走完 Track 1（Ampere）、Track 2 的大部分内容需要 H100/云实例，Track 3 中只有 3.2 节（FP4/FP6 精度）能在 RTX 5070 上直接实践，3.1/3.3/3.4 建议先理解原理（配合本文交叉引用的深度章节），有条件时再上云验证。
