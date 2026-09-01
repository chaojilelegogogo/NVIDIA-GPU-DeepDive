# 第二部分 CUDA 编程模型（软件如何映射到硬件）

> 前提：你已经理解了第一部分的硬件结构。这一部分的任务是把 Grid/Block/Warp/Thread、Memory Hierarchy 这些"物理事实"，翻译成"软件工程师能写代码的抽象接口"。读完这一部分，你应该能明确回答：**为什么 Block 不能跨 SM？为什么 Warp 固定 32 线程？为什么 Shared Memory 属于 Block？**——这三个问题在第一部分已经给出了硬件层面的答案，这一部分要把它们落实到具体的编程模型规则和 API 语义上。

## 第二章 CUDA Programming Model

### 2.1 Grid：一次 Kernel 启动的全部工作量

Grid 是一次 `kernel<<<...>>>()` 调用所产生的全部线程的集合，在逻辑上组织成 1D/2D/3D 的 Block 数组。Grid 的维度（`gridDim`）由程序员在启动时指定，代表"这个计算任务总共需要多少个 Block 来完成"。

Grid 层面的关键设计事实：**Grid 内的 Block 之间原则上没有执行顺序保证，也没有硬件级别的同步原语（Cooperative Groups 的 Grid Sync 是软件+驱动层面的特例，见第七部分）**。这不是疏忽，而是有意为之——只有当 Block 之间完全独立、执行顺序无关紧要时，GPU 的 Block 调度器（GigaThread Engine）才能自由地把 Block 动态分发到任意空闲 SM 上，从而让同一份 kernel 代码在拥有 20 个 SM 的小 GPU 和拥有 132 个 SM 的大 GPU 上都能正确运行，只是并行度和总耗时不同——这就是 CUDA 编程模型标榜的**透明可扩展性（Transparent Scalability）**。

### 2.2 Block：能够互相通信的最大线程集合

Block（Thread Block，硬件/PTX 层面称为 CTA，Cooperative Thread Array）是 CUDA 编程模型中最重要的一个抽象层级，因为它精确对应第一部分讲的一个硬件事实：**一个 Block 会被完整调度到一个 SM 上，且在其生命周期内始终驻留在该 SM，不会被迁移，也不会跨 SM 拆分执行。**

#### 2.2.1 为什么 Block 不能跨 SM？（正式回答）

答案由三个互相支撑的硬件事实共同决定：

1. **Shared Memory 是 SM 的私有物理资源**。它是一块物理上焊在 SM 内部的 SRAM，不存在一种硬件机制能让两个不同 SM 的计算单元以低延迟访问同一块 Shared Memory。如果 Block 被允许跨 SM，那么 `__shared__` 变量就无法用一份统一、低延迟的物理存储来实现。
2. **`__syncthreads()` 依赖 SM 内部的硬件 Barrier 单元**。Block 内同步用的硬件屏障电路是 SM 局部的，跨 SM 做同步需要通过 L2/Global Memory 这种高延迟路径，代价比片内同步高几个数量级，会摧毁 Block 内同步"轻量、高频"的设计假设。
3. **Warp Scheduler 和寄存器堆的资源分配以 SM 为单位**。一个 Block 需要的寄存器、Shared Memory、Warp 槽位都是在 Block 被分发到某个 SM 时，一次性从该 SM 的资源池中"预留"出来的（占用率计算就是基于这个事实，见 2.8 节）。如果 Block 可以跨 SM，这种"整块预留资源"的简单模型就会崩溃，硬件调度会复杂得多。

（注：Hopper 引入的 Thread Block Cluster 没有打破“单个 Block 属于单个 SM”这条基本规则；它是在这条规则之上叠加的可控协作范围。执行模型见 [2.2.4](#224-thread-block-clusterhopper)；DSM 见[第三部分 3.10](./part03-cuda-memory/08-cluster-dsm-tmem.md)；`cluster.sync()` 见[第十三部分](./part13-synchronization-handbook/01-execution-barriers.md)。）

#### 2.2.2 Block 内的通信与同步原语

- `__shared__` 变量：Block 内所有线程共享，生命周期与 Block 相同。
- `__syncthreads()`：Block 级别的硬件屏障，保证所有线程都执行到这个点之后才能继续，且保证屏障之前的所有 Shared Memory/Global Memory 写入对其它线程可见。

#### 2.2.3 Block 的维度与线程索引

Block 内部同样可以组织成 1D/2D/3D（`blockDim.x/y/z`），这只是为了让程序员在处理图像、矩阵、体数据等多维数据结构时索引更直观，硬件在执行层面只关心"Block 内线程被线性化后的顺序"（`threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x`），并按照这个线性顺序，每连续 32 个线程打包成一个 Warp。

#### 2.2.4 Thread Block Cluster（Hopper）

Hopper 把执行层级从四层扩展为五层：

```text
传统：
  Thread → Warp → Block(CTA) → Grid

Hopper+：
  Thread → Warp → Block(CTA) → Cluster → Grid
```

**为什么 Block 之间长期不能协作？** 2.2.1 已说明：默认 Grid 中的 Block 独立调度、无硬件级会合，才能换来透明可扩展性。代价是：

1. 相邻 tile 想复用同一块 shared 热数据时，只能经 Global/L2 绕行；
2. Tensor Core 吞吐上升后，跨 CTA 的数据复用成为瓶颈；
3. Cooperative Launch 的 `grid.sync()` 虽能跨 Block 会合，但要求**整个 Grid 同时驻留**，Grid 尺寸被设备 occupancy 硬卡住，不适合作为默认协作模型。

**Hopper Cluster Execution Model** 提供介于 CTA 与 Grid 之间的一层：一组 CTA 被**共同调度**到可低延迟互联的硬件范围（通常是一组相邻 SM），从而获得：

| 能力 | 含义 | 本部分是否深讲 |
|---|---|---|
| 共同驻留 | cluster 内 CTA 可假定彼此同时存在 | ✅ 执行模型 |
| `clusterDim` / `block_rank` | 逻辑坐标与 cluster 内 CTA 序号 | ✅ 编程语义 |
| `cluster.sync()` | cluster 范围 execution barrier | 指向 Part 13 |
| DSM / `map_shared_rank` | 访问 peer CTA 的 shared | 指向 Part 03 |
| TMA multicast | 一次 load 写入多个 CTA shared | 指向 Part 03 |

声明方式（示意）：

```cuda
__global__ void __cluster_dims__(2, 1, 1) kernel(...);

// 或 launch 时：
// cudaLaunchAttributeClusterDimension → clusterDim = (2,1,1)
```

Cluster 内常用标识：

| 概念 | 含义 |
|---|---|
| `clusterDim` | 一个 cluster 含多少个 CTA（如 2×1×1） |
| `block_rank` | 当前 CTA 在 cluster 内的序号（0 … num_blocks-1） |
| `gridDim` / `blockIdx` | 仍描述整个 Grid；cluster 是其上的额外调度分区 |

**Cluster scheduling**：GigaThread 不再把这些 CTA 当作完全独立的工作项随机分发，而要满足“同一 cluster 的 CTA 能同时落到可通信的 SM 集合”。因此：

- 合法 cluster 大小受设备属性约束（portable / non-portable cluster size）；
- 过大的 cluster 会降低可调度性，可能降低占用；
- **未**以 cluster 方式 launch 的普通 kernel，不能调用 `this_cluster()` 并假定 DSM 可用。

本部分只回答“软件执行层级如何映射硬件调度”。DSM 地址与性能见[第三部分 3.10](./part03-cuda-memory/08-cluster-dsm-tmem.md)；`cg::this_cluster()` API 见[第七部分 7.4](./part07-cuda-cpp-api.md)；`cluster.sync()` 与 DSM 可见性见[第十三部分 13.1.3](./part13-synchronization-handbook/01-execution-barriers.md)。

### 2.3 Thread：最小的逻辑执行单元

每个 Thread 拥有：私有寄存器、私有的 Local Memory（寄存器溢出时使用，物理上位于片外，详见第三部分）、私有的程序计数器（Volta 起独立）、通过内建变量 `threadIdx`/`blockIdx`/`blockDim`/`gridDim` 可以计算出自己的全局唯一 ID，从而决定自己该处理哪一份数据——这是 SIMT 编程模型下"用同一份代码处理不同数据"的核心手法。

### 2.4 Warp：硬件真正的调度和执行单位

再次强调这个跨越 Part 1/Part 2 的核心事实：**Warp 是编程模型里"看不见但决定一切"的隐藏单位**——CUDA C++ 语法层面没有 `warp` 这个关键词来声明或创建 Warp，Warp 完全是由 Block 内线程按连续 32 个一组自动划分出来的，但几乎所有和性能相关的行为（分支分歧、内存合并访问、Warp 级原语、Tensor Core 编程）都是以 Warp 为单位发生的。

#### 2.4.1 为什么 Warp 固定 32 线程？（编程模型视角的补充回答）

第一部分已经从硬件权衡角度回答过这个问题。从编程模型角度补充一点：正因为 Warp 大小历史上稳定不变（`warpSize` 内建变量恒为 32），CUDA 生态才能安心地把大量优化（如 Warp 级归约模板、`shfl.sync` 蝶式交换模式、Tensor Core 的 `m16n8k*` 操作数分布）硬编码为"32"这个常数，而不需要处理"可变 Warp 宽度"这种在移植性和性能上都更复杂的场景（对比 AMD GPU 的 Wavefront 有 32/64 两种宽度，需要软件适配）。

### 2.5 Kernel Launch：`<<<Grid, Block>>>` 背后发生了什么

一次 kernel 启动（`kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args)`）在硬件/驱动层面大致经历：

1. Runtime 把 Grid/Block 维度、共享内存大小、Stream 归属、Kernel 参数打包，通过 Driver API 提交到硬件的命令队列。
2. GigaThread Engine（全局调度器）根据这次启动需要的 Block 数量，以及每个 SM 上剩余的资源（寄存器、Shared Memory、Block 槽位、Warp 槽位），开始把 Block **逐个动态分发**给当前有空闲资源的 SM。
3. 每个 SM 接收到一个 Block 后，为其一次性分配好所需的寄存器和 Shared Memory 空间，并把它拆分成若干 Warp 放入本地的 Warp 调度队列。
4. 一旦一个 Block 的所有线程都执行完毕并退出，该 Block 占用的资源被释放，SM 变得有空闲资源，GigaThread Engine 可以再分发一个新 Block 过来——这就是为什么**一个 SM 在其生命周期内可能会串行处理多个 Block**（Block 数量超过 SM 数量时）。

### 2.6 Stream：GPU 上的异步任务队列

Stream 是 CUDA 里实现**任务级并行/异步执行**的核心抽象：同一个 Stream 内的操作（Kernel 启动、内存拷贝）保证按提交顺序串行执行；不同 Stream 之间的操作**没有顺序保证，硬件会尽量并发执行它们**（受限于实际的硬件资源，如 Copy Engine 数量、SM 是否还有空闲资源）。

Stream 存在的意义在于解决一个具体问题：如果所有操作都在默认 Stream（Stream 0）里顺序执行，那么"CPU→GPU 拷贝数据"、"GPU 计算"、"GPU→CPU 拷贝结果"这三步永远串行，GPU 的计算单元和拷贝引擎（Copy Engine，独立于 SM 的 DMA 硬件）在等待彼此时都被浪费。用多个 Stream，可以让"计算 Batch N 的同时，拷贝 Batch N+1 的数据"，这是发挥硬件里独立 DMA 引擎和 SM 计算能力可以并发这一事实的编程手段。这个"双缓冲/流水线"思想会在第四部分的 Pipeline 编程和第三部分的异步拷贝技术中反复出现——本质上是同一个思想在不同粒度（Host-Device 级 vs Warp 内部级）上的体现。

### 2.7 Event：GPU 时间线上的时间戳与依赖点

Event 是插入到 Stream 时间线中的一个"标记点"，有两大用途：

1. **精确计时**：`cudaEventRecord` 前后各插入一个 Event，用 `cudaEventElapsedTime` 计算 GPU 上真实的执行耗时（比 CPU 端 `clock()` 计时更准确，因为 GPU 执行本身是异步的，CPU 端计时会把排队等待时间也算进去）。
2. **跨 Stream 依赖**：`cudaStreamWaitEvent` 可以让一个 Stream 等待另一个 Stream 中某个 Event 完成后才继续，从而在"默认所有 Stream 互相独立"的模型上，精确地插入必要的依赖关系，而不必粗暴地用一个全局同步（`cudaDeviceSynchronize`）打断所有并发性。

### 2.8 Occupancy（占用率）：为什么这是资源分配问题而不是"调优参数"

Occupancy 定义为：**一个 SM 上实际驻留的 Warp 数，除以该 SM 硬件支持的最大 Warp 数**。它不是一个孤立的调优旋钮，而是三种硬件资源同时受限的结果：

```
一个 SM 能同时驻留多少个 Block，由以下三者中最紧的那个决定：

1. 寄存器限制： floor(SM总寄存器数 / (每Block线程数 × 每线程寄存器数))
2. Shared Memory限制： floor(SM总SharedMemory / 每Block使用的SharedMemory)
3. Block/Warp数量限制： SM硬件规定的最大并发Block数、最大并发Warp数（架构固定值）

实际驻留Block数 = min(以上三者)
```

这就是为什么"减少寄存器使用量"、"减少 Shared Memory 用量"能提高 Occupancy——本质上是在让每个 Block "变瘦"，从而让 SM 的固定资源池能塞下更多 Block、驻留更多 Warp。

但**高 Occupancy 不等于高性能**，这是一个常被误解的点：Occupancy 只是"提供了更多可供 Warp Scheduler 选择、用来隐藏延迟的候选 Warp"，如果 kernel 本身不是延迟受限型（比如是 Compute Bound 且已经有很好的指令级并行度），盲目提高 Occupancy（比如以牺牲每线程寄存器数、导致寄存器溢出到 Local Memory 为代价）反而会降低性能。这个话题会在第十一部分结合具体案例展开。

### 2.9 Memory Model：编程模型看到的存储空间

CUDA 编程模型把第一部分讲的物理存储层次，包装成程序员可以用 C++ 语法直接声明和访问的几种存储类别（详细的硬件实现、性能特性放在第三部分整章展开，这里先建立编程模型层面的分类）：

| 存储类别 | 声明方式 | 作用域 | 生命周期 |
|---|---|---|---|
| Register | 普通局部变量（编译器自动分配） | Thread | Kernel 执行期间 |
| Local Memory | 寄存器溢出/大数组局部变量 | Thread | Kernel 执行期间 |
| Shared Memory | `__shared__` | Block；Hopper+ 经 DSM 可被同 cluster 的 peer CTA 访问 | Block 执行期间（peer 访问期间 owner 不得退出） |
| Global Memory | `cudaMalloc` 分配的指针 | Grid（全局） | 显式释放前 |
| Constant Memory | `__constant__` | Grid（全局，只读） | 程序运行期间 |
| Texture/Surface | Texture Object API | Grid（全局，只读/读写） | 显式创建/销毁 |

编程模型层面的关键点是**作用域即约束**：一个 `__shared__` 变量的作用域被语言层面限定为 Block，这不是语法上随意的选择，而是精确反映了 2.2 节讲的"Shared Memory 是 SM 私有物理资源、只能被驻留在该 SM 上的这一个 Block 访问"这一硬件事实——**为什么 Shared Memory 属于 Block？答案是：因为语言的作用域规则如实映射了硬件的物理归属关系，而不是反过来。**

### 2.10 Synchronization：编程模型提供的同步原语层级

CUDA 的同步原语精确对应第一部分讲到的硬件层级，级别越高，代价越大：

```
__syncwarp(mask)      → Warp 内同步，硬件开销最低（几个周期）
__syncthreads()       → Block 内同步，SM 内硬件 Barrier（几十周期量级）
cluster.sync()        → Cluster 内同步（Hopper+），保护 DSM/跨 CTA 交接；需 cluster launch
grid.sync()           → Cooperative Grid 同步，要求整个 Grid 同时驻留
cudaStreamSynchronize → Stream 级同步，CPU 等待 GPU 某个 Stream 排空
cudaDeviceSynchronize → 整个 Device 级同步，代价最高
```

同步原语的分层设计再次体现了"编程模型如实反映硬件通信代价"这一贯穿全书的思想：**能用更细粒度的同步解决的问题，就不要用更粗粒度的同步**。Cluster sync 比 `grid.sync()` 便宜、比 `__syncthreads()` 贵，且只在声明了 cluster 的 kernel 中可用。

> **同步语义提示**：执行会合、内存可见性和异步操作完成不是同义词。`__syncthreads()` / `cluster.sync()` 是 execution barrier；`__threadfence*()` 不会等待其它线程；`cp.async`/TMA/WGMMA 的完成需要各自的 wait 协议。完整映射见[第十三部分](./part13-synchronization-handbook/00-overview.md)。

---

### 本部分小结：三个问题的正式解答

1. **为什么 Block 不能跨 SM？** 因为 Shared Memory 是 SM 的私有物理 SRAM，`__syncthreads()` 依赖 SM 内部的硬件屏障电路，且 SM 的寄存器/Shared Memory 资源是以整个 Block 为单位一次性预留的——这三者共同要求一个 Block 的全部线程必须运行在同一个 SM 上。
2. **为什么 Warp 固定 32 线程？** 这是 NVIDIA 在"指令发射效率"（Warp 越大，摊薄译码/发射开销的收益越大）与"分支分歧代价"（Warp 越大，Warp 内至少出现一次分支分歧的概率越高，效率损失越大）之间，经过多代硬件验证后选定的工程权衡值，并且为了生态稳定性，长期保持不变。
3. **为什么 Shared Memory 属于 Block？** 因为它在硬件上物理属于 SM，而一个 Block 恰好被完整绑定在一个 SM 上执行——编程模型的作用域规则只是"如实转译"了这个硬件事实，而不是一个独立的语言设计决策。Hopper DSM 允许同 cluster 的 peer CTA **远程访问**这份仍属于 owner CTA 的 shared，并不把 shared 改成“Cluster 私有统一 SRAM”。

4. **为什么还要 Cluster？** 因为默认 Block 独立调度无法低成本复用相邻 tile；Cooperative Grid 又太重。Cluster 在不牺牲“单 Block 仍绑定单 SM”的前提下，给出可控的跨 SM 协作范围。

下一部分（第三部分，CUDA Memory）将是全书篇幅最大、也是现代高性能 CUDA 开发中最重要的一部分——因为随着 Tensor Core 算力的指数级增长（第五部分），"如何以更少的代价把数据喂给计算单元"已经取代"如何多做算力"，成为决定 kernel 性能上限的第一因素。Cluster 之上的 DSM 地址与性能模型在第三部分 3.10 展开。
