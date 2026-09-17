# 第十部分 GPU 硬件发展史（整个课程的灵魂）

> 这一部分建议在读完 Part 1~2 后先通读一遍（建立时间线），学完 Part 3~9 后再精读一遍（这时每个名词背后都有具体技术细节支撑）。每一代架构，固定回答四个问题：**增加了什么？为什么增加？软件怎么支持？性能提升在哪里？** 把这条时间线串起来，你会发现 NVIDIA 三十年的架构演进其实只在解决一个不断被重新表述的问题：**如何让海量、越来越强的算力，被高效、低开销地"喂饱"**。

## 10.0 总览时间线

```
Tesla(2006) → Fermi(2010) → Kepler(2012) → Maxwell(2014) → Pascal(2016)
   → Volta(2017) → Turing(2018) → Ampere(2020) → Hopper(2022) → Blackwell(2024)
```

本章从 Fermi 开始（第一个具有现代意义上完整"通用计算"能力的架构，也是 CUDA 生态真正成型的起点），重点从 Volta 开始详细展开（Tensor Core 时代），因为这之后的每一代都直接构成第五部分的核心内容。

## 10.1 Fermi（2010，SM 20）—— 通用计算的奠基

- **增加了什么**：统一的 L1/L2 Cache 层次（此前 GPU 几乎没有真正意义的通用数据 Cache，主要靠 Shared Memory 和纹理缓存）；真正的双精度（FP64）性能大幅提升；支持 C++ 特性（虚函数、异常处理的部分子集）；ECC 内存纠错（面向数据中心可靠性需求）；并发 Kernel 执行（多个 Kernel 可以同时跑在同一个 GPU 上，不必严格排队）。
- **为什么增加**：GPU 在 Fermi 之前主要还是"图形硬件顺带能做点通用计算"，Fermi 是 NVIDIA 第一次系统性地把 GPU 当作**通用并行计算平台**来设计——引入 Cache 层次是因为通用计算负载（不像图形渲染那样访问模式高度规整可预测）需要更"像 CPU"的透明缓存来兜底不规则访问；双精度和 ECC 则是为了打入高性能科学计算市场。
- **软件怎么支持**：CUDA 编程模型（Grid/Block/Warp/Thread）在这一代基本定型，后续架构都是在此基础上扩展，而不是推倒重来。
- **性能提升在哪里**：通用计算负载（不只是规整的图形/矩阵运算）第一次能在 GPU 上获得有意义的加速比，是 GPGPU（General-Purpose GPU）这个概念真正大规模产业化的起点。

## 10.2 Kepler（2012，SM 30/35）—— 能效与动态并行

- **增加了什么**：`shfl`（Warp Shuffle，第六部分 6.1.1）指令首次出现；动态并行（Dynamic Parallelism，Kernel 内部可以直接启动新的子 Kernel）；Hyper-Q（允许多个 CPU 线程/进程同时向 GPU 提交任务队列，减少排队等待）；更高的能效比（每瓦性能显著提升）。
- **为什么增加**：`shfl` 解决的问题正是第六部分讲的——在此之前 Warp 内线程交换数据必须经过 Shared Memory，Kepler 提供了直接的寄存器级数据交换通路；动态并行则是为了支持"计算量在运行时才能确定"的不规则并行任务（如自适应网格细化），不必每次都退回 CPU 端做任务分解再重新发起 Kernel。
- **软件怎么支持**：`__shfl()` 系列 intrinsic；`cudaLaunchKernel` 可以在 Device 代码中被调用。
- **性能提升在哪里**：Warp 内归约类算法（reduce/scan）因为 Shuffle 指令大幅减少了 Shared Memory 访问和同步开销；数据中心场景下能效比提升带来了大规模部署的成本优势。

## 10.3 Maxwell（2014，SM 50）—— 能效架构的极致化

- **增加了什么**：进一步细化 SM 内部分区结构（更小、更多的 Processing Block，每个分区独立的 Warp Scheduler），改进的调度和指令发射效率；更大的 Shared Memory 配置灵活性；显著提升的每瓦性能（虽然制程和 Kepler 相近，但架构效率大幅提升）。
- **为什么增加**：这一代架构的核心目标非常明确——在制程工艺没有大幅跃进的情况下，纯靠架构层面的效率优化挤出性能和能效提升，把 SM 拆分成更小的独立调度分区，减少每个分区内部资源争用，是这一思路的直接体现。
- **软件怎么支持**：编程模型层面变化不大，主要是 `ptxas` 针对新的 SM 分区结构调整指令调度和寄存器分配策略。
- **性能提升在哪里**：同等功耗预算下的性能密度显著提升，是很多社区手工 SASS 优化项目（如著名的 maxas，对 Maxwell SGEMM 手工调度 Reuse Flag/Stall Count 达到接近理论峰值的性能）选择的研究对象，从侧面说明这一代架构的指令级调度对性能的影响非常显著、且有很大的手工优化空间。

## 10.4 Pascal（2016，SM 60）—— 大容量高带宽内存与 NVLink 元年

- **增加了什么**：首次搭载 **HBM2**（高带宽内存，通过 CoWoS 等 2.5D 封装technology），显存带宽大幅跃升；**NVLink**（GPU 间高速互联，带宽远超传统 PCIe）首次引入；统一内存（Unified Memory）获得硬件级页错误处理支持（此前更多依赖软件模拟）；原生 FP16 计算支持增强。
- **为什么增加**：随着深度学习训练规模的扩大，单卡显存带宽和多卡互联带宽逐渐成为大规模训练的明显瓶颈——这是第一部分 Roofline 模型讲的"算力增长快于带宽增长"这一矛盾第一次在产品层面被正面回应：先用 HBM2 大幅提升单卡访存带宽，同时用 NVLink 解决"多卡协同训练时，卡间通信比卡内计算慢得多"的新瓶颈。
- **软件怎么支持**：`cudaMallocManaged` 的 Unified Memory 编程模型因硬件页错误处理支持而变得更实用（不再需要频繁的显式预取提示）；NCCL 等多卡通信库开始利用 NVLink 拓扑。
- **性能提升在哪里**：显存带宽敏感型 kernel（很多深度学习负载本身就是 Memory Bound，参考第一部分 Roofline）获得直接的带宽红利；多卡分布式训练的扩展效率显著提升。

## 10.5 Volta（2017，SM 70）—— Tensor Core 元年 + 独立线程调度

- **增加了什么**：**Tensor Core**（详见第五部分 5.3）；**Independent Thread Scheduling**（独立线程调度，第一部分 1.4.4）；`mma.sync`/`wmma` 编程接口；更大的 Shared Memory（96KB/SM）与合并的 L1/Shared 物理存储；NVLink 2.0。
- **为什么增加**：Tensor Core 的出现是整本教程反复强调的分水岭——深度学习负载几乎全部由矩阵乘法主导，用通用 CUDA Core 逐元素做乘加，在面积/功耗预算下已经无法跟上算力需求增长，NVIDIA 选择了"领域专用架构"路线，牺牲部分通用性换取数量级的矩阵乘法吞吐提升（详见第五部分 5.2）。独立线程调度则是为了修复 SIMT 模型中"Warp 内线程无法灵活互相等待/通信"这一长期存在的局限（第一部分 1.4.4 已详述具体动机）。
- **软件怎么支持**：CUDA 9 引入 `wmma` C++ API 与 Cooperative Groups；CUDA 10.1 开放更底层的 `mma.sync` PTX 指令；`__shfl`/`__ballot` 等 Warp 内建函数全面加上 `_sync` 后缀。
- **性能提升在哪里**：深度学习训练/推理中的矩阵乘法/卷积核心运算获得数量级的吞吐提升；依赖细粒度线程间通信的并行算法（如某些图算法、锁自由数据结构）因独立线程调度获得了正确性和性能上的双重改善。

## 10.6 Turing（2018，SM 75）—— 推理场景的低精度扩展

- **增加了什么**：Tensor Core 支持 INT8/INT4（详见第五部分 5.4）；RT Core（光线追踪硬件加速单元，主要面向图形渲染，通用计算范畴外，此处不展开）；进一步改进的 L1/Shared Memory 结构。
- **为什么增加**：深度学习模型开始大规模落地"推理"场景（相比训练，推理对数值精度容忍度更高，但对延迟/吞吐更敏感），INT8/INT4 用更小的数据位宽换取更高的算力密度和更低的访存压力，专门服务这一新兴的重要场景。
- **软件怎么支持**：TensorRT 等推理框架深度利用 INT8 Tensor Core；`mma.sync` 指令族扩展支持整数类型操作数。
- **性能提升在哪里**：推理吞吐（尤其是 batch 较大、对延迟不极端敏感的服务端推理场景）大幅提升，同等硬件成本下能服务更多并发推理请求。

## 10.7 Ampere（2020，SM 80）—— 数据搬运机制的第一次系统性升级

- **增加了什么**：`cp.async` 异步拷贝指令（详见第三部分 3.8）；Tensor Core 第三代，支持 **TF32/BF16**、结构化稀疏加速（2:4 稀疏矩阵可获得额外吞吐提升）；`ldmatrix` 指令（第五部分 5.4）；更大更灵活的 L2 Cache 及其常驻（Persisting）区域控制；多实例 GPU（MIG，把一张物理 GPU 硬件级切分成多个独立的小 GPU 实例）。
- **为什么增加**：这是本教程第三部分开篇就点出的核心矛盾第一次被正面命名——**Shared Memory Load（准确地说是"用 Warp 参与、经过寄存器中转的 Global→Shared 搬运"）成为瓶颈**：Volta/Turing 的 Tensor Core 已经把矩阵乘法算力提得很高，但"喂数据"这一步仍然占用宝贵的寄存器和 Warp 发射带宽，`cp.async` 正是为了直接切掉这一层开销而生。TF32/BF16 则是训练场景在数值范围和速度之间寻求更好平衡点的直接产物（第五部分 5.4 详述）。
- **软件怎么支持**：`cuda::memcpy_async`（`<cuda/pipeline>`，第七部分 7.5）、`cooperative_groups::memcpy_async`；PTX `cp.async.ca`/`cp.async.cg` + `commit_group`/`wait_group`。
- **性能提升在哪里**：数据搬运与计算能在软件流水线（第四部分 Double/Triple Buffer）中更彻底地重叠，减少了搬运阶段对 Warp 计算资源的占用；TF32 让大量此前顾虑数值稳定性、不敢轻易切换到 FP16 的训练任务，几乎零成本获得数倍加速。

## 10.8 Hopper（2022，SM 90）—— 全异步 GPU：TMA + Warp Group + Cluster

- **增加了什么**：**TMA**（Tensor Memory Accelerator，第三部分 3.9）；**Warp Group 与 WGMMA**（第五部分 5.5）；**Thread Block Cluster 与 Distributed Shared Memory (DSM)**（第三部分 3.10）；异步事务屏障 `mbarrier`（第六部分 6.2.2）；Tensor Core 第四代原生支持 **FP8**；`setmaxnreg` 动态寄存器重分配（支撑 Warp Specialization，第四部分 4.8）。
- **为什么增加**：`cp.async` 虽然实现了异步，但仍然是"**Warp 参与搬运**"——每个线程仍要各自计算地址、各自发起指令，没有利用"要搬的是一整块规整张量"这一结构化信息。TMA 把这部分工作彻底转移给专用硬件电路，只需一个线程发起（第三部分 3.9 已详细展开）。与此同时，Ampere 级别的 `mma.sync` 受限于"Warp（32线程）"这个协作颗粒度，难以进一步扩大处理的 tile 规模来匹配 Hopper 大幅提升的 Tensor Core 峰值算力，于是有了 Warp Group（128线程协作）与 WGMMA（第五部分 5.5）。Thread Block Cluster 则是在"Block 不能跨 SM"（第二部分 2.2.1）这条基本规则**之上**，新增了一层允许多个 Block（跨 SM）通过 Distributed Shared Memory 直接互访彼此 Shared Memory 的协作机制，进一步扩大了"能够高效通信的线程集合"的边界，为 TMA Multicast、跨 SM 协作 GEMM 等技术提供了硬件基础。
- **软件怎么支持**：`cuTensorMapEncodeTiled` 构建 TMA 描述符；`cp.async.bulk.tensor.*` PTX 指令族；`wgmma.mma_async` + `fence`/`commit_group`/`wait_group` 协议；`cluster.sync()`（Cooperative Groups 对 Cluster 同步的封装）；CUTLASS 3.x/CUTE 是这一代架构上事实标准的高性能库编程范式（第十二部分详细分析）。
- **性能提升在哪里**：GEMM/Attention 类核心算子在 Hopper 上通过 TMA+WGMMA+Warp Specialization 的组合，可显著提高 Tensor Core 利用率；实际数值取决于 shape、layout、stage 数、Toolkit 和测量方法（第五部分 5.5）。

## 10.9 Blackwell（2024，SM 100/SM 120）—— 累加器搬出寄存器：TMEM 与极致低精度

- **增加了什么**：**第五代 Tensor Core**，原生支持 **FP4/FP6**（NVFP4 微块缩放格式，第五部分 5.6）；**TMEM**（Tensor Memory，第三部分 3.11、第五部分 5.6）；`tcgen05` 指令族（`mma`/`ld`/`st`/`cp`/`alloc`）；**CTA Pair / Cluster Tensor**（TPC 内相邻两 SM 协作共享操作数，第五部分 5.6）；第二代 Transformer Engine；硬件解压缩引擎（面向压缩模型权重的加载场景）；进一步扩展的 NVLink（第五代）支持更大规模的多 GPU 直连拓扑（如 GB200 NVL72 机架级设计）。
- **为什么增加**：Hopper 的 WGMMA 已经把协作颗粒度和数据来源问题解决得很好，但**累加器（Accumulator）本身还在寄存器堆里**——随着 Tensor Core 处理的 tile 进一步增大、精度进一步降低（意味着单位存储能装下更多元素、单次矩阵乘法产生的累加结果规模相应增大），256KB/SM 这个多代未变的寄存器堆容量逐渐吃紧，成为继续放大 tile、提升算力利用率的新瓶颈——这正是"寄存器放不下，所以增加 Tensor Memory"这条第三部分/第五部分反复强调的因果链的最终落点。同时，精度维度下探到 4/6 bit 后，朴素量化误差已经不可忽视，必须由硬件级微块缩放机制（NVFP4）来兜底可用精度。
- **软件怎么支持**：PTX/SASS 新增 `tcgen05.*` 指令族；累加结果的读写不再直接对应寄存器变量，而是需要显式的 TMEM 分配/寻址（`tcgen05.alloc`），编程模型上更接近 Shared Memory 而不是寄存器（第三部分 3.11 已详述）；CUTLASS 3.x 后续版本、cuBLAS/cuDNN 内部逐步跟进适配 Blackwell 专属的 `tcgen05` 路径。
- **性能提升在哪里**：极大规模的 GEMM/Attention（尤其是训练/推理中 Transformer 架构的核心算子）在 Blackwell 上能利用更大的 tile 尺寸和更低精度获得数倍于 Hopper 的有效吞吐（尤其是 FP4/FP6 推理场景）；CTA Pair 机制减少了相邻 SM 间的冗余数据搬运，进一步压榨访存带宽的利用效率。
- **注意（消费级 vs 数据中心）**：以上 TMEM/`tcgen05`/CTA Pair 特性**只存在于数据中心 Blackwell**（B100/B200/B300，`sm_100a`/`sm_103a`）。GeForce RTX 50 系列消费级 Blackwell（`sm_120`）不具备 TMEM 硬件，Tensor Core 仍走 Hopper 式的寄存器/Shared Memory 路径（`mma.sync.aligned.block_scale`），详见第三部分 3.11 节与第五部分 5.6 节的专门说明。

## 10.10 全景对照表

| 架构 | 年份 | 计算能力 | 最关键的新增特性 | 解决的核心瓶颈 |
|---|---|---|---|---|
| Fermi | 2010 | SM 20 | 通用 L1/L2 Cache、C++ 支持、并发 Kernel | GPU 从图形硬件走向通用计算平台 |
| Kepler | 2012 | SM 30/35 | `shfl`、动态并行、Hyper-Q | Warp 内通信开销、能效、任务提交排队 |
| Maxwell | 2014 | SM 50 | 更细粒度 SM 分区、调度效率优化 | 制程停滞下的架构级能效提升 |
| Pascal | 2016 | SM 60 | HBM2、NVLink、硬件级 Unified Memory | 单卡带宽 + 多卡互联带宽 |
| Volta | 2017 | SM 70 | **Tensor Core**、独立线程调度 | 矩阵乘法算力密度、Warp 内灵活通信 |
| Turing | 2018 | SM 75 | Tensor Core INT8/INT4 | 推理场景的算力/能效 |
| Ampere | 2020 | SM 80 | **`cp.async`**、TF32/BF16、`ldmatrix` | Warp 参与搬运挤占计算资源 |
| Hopper | 2022 | SM 90 | **TMA**、Warp Group/WGMMA、Cluster/DSM、FP8 | 逐线程地址生成开销、协作颗粒度不足 |
| Blackwell | 2024 | SM 100/120 | **TMEM**、`tcgen05`、FP4/FP6、CTA Pair | 累加器挤占寄存器堆、极致低精度可用性 |

## 10.11 这条时间线揭示的第一性原理

把这张表放在一起看，会发现三条并行推进、互相交织的主线：

1. **算力主线**：CUDA Core（标量）→ Tensor Core（矩阵专用电路，Volta 起）→ 精度持续下探（FP16→INT8→TF32/BF16→FP8→FP4/FP6）→ 协作颗粒度持续扩大（Warp→Warp Group→跨 SM CTA Pair）。
2. **搬运主线**：`ld.global`+`st.shared`（线程搬）→ `cp.async`（Warp 异步搬）→ TMA（专用硬件 DMA 搬）→ TMEM 配套的 `tcgen05.cp`（结果直接落地专属存储，减少搬运本身的必要性）。
3. **协作范围主线**：Thread → Warp（32线程，SIMT 基本颗粒度）→ Block（SM 内可同步/通信的最大范围）→ Warp Group（Hopper，128线程 Tensor Core 协作颗粒度）→ Thread Block Cluster / CTA Pair（跨 SM，Hopper/Blackwell）。

这三条主线其实是同一个根本矛盾在不同维度的投影：**算力（第一条主线）的增长速度，持续快于片上/片外数据搬运带宽和线程协作带宽（第二、三条主线）的增长速度**。每一代新增的特性，几乎都可以归类为"给算力主线的最新进展，找到一种同等量级的搬运/协作效率提升方案"。理解了这一点，你面对任何一个新指令、新硬件单元，都可以先问："它是算力侧的新增强，还是搬运/协作侧针对上一代算力增强所做的适配？"——这个问题几乎总能帮你迅速定位这个新特性存在的意义,也是预判 Rubin、Feynman 等未来架构演进方向的最可靠工具。

带着这条完整的因果链，我们进入最后两部分——先系统化性能优化的实战方法论（第十一部分），再落地到具体的工业级 kernel 源码分析（第十二部分），把整本教程建立的所有硬件直觉，真正应用到读代码、写代码的实践中去。
