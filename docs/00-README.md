# AI 软硬件全栈教程（NVIDIA GPGPU / CUDA 高性能内核开发 → 分布式训练与推理系统）

> 定位：不是"CUDA API 大全"，而是**"硬件发展驱动软件演进"**的因果链教程——从单卡微架构一路讲到万卡集群上的训练与推理系统。
> 目标读者：希望成为 Blackwell 时代高性能 CUDA 工程师、能读懂 CUTLASS / FlashAttention / cuBLAS / cuDNN / NCCL / vLLM 源码的人。
> 学习周期建议：卷一~卷二 4~6 个月（全职或每天 2~3 小时）；卷三~卷七作为后续扩展。
> 状态：**卷一~卷二（Part 1~13）已完成初稿**，正文合计约 20 万字符；**卷三~卷七（Part 14~20）已建目录骨架，内容待填充**。

## 全库分卷结构

| 卷 | 主题 | 包含 Part | 状态 |
|---|---|---|---|
| **卷一** [vol1-gpu-hardware/](vol1-gpu-hardware/) | GPU 硬件与体系结构 | Part 1 / 4 / 10 + 型号手册（A100/H100/B200）+ roadmap | ✅ 初稿完成 |
| **卷二** [vol2-cuda-software/](vol2-cuda-software/) | CUDA 软件栈与 Kernel 工程 | Part 2 / 3 / 5 / 6 / 7 / 8 / 9 / 11 / 12 / 13 | ✅ 初稿完成 |
| **卷三** [vol3-interconnect-communication/](vol3-interconnect-communication/) | 互联与通信（NVIDIA 生态为主） | Part 14 互联硬件 / Part 15 集合通信 | Part 15 ✅ 初稿；Part 14 🚧 骨架 |
| **卷四** [vol4-distributed-training/](vol4-distributed-training/) | 分布式训练系统 | Part 16 并行策略 / Part 17 框架实现 | 🚧 骨架 |
| **卷五** [vol5-ai-compilers/](vol5-ai-compilers/) | AI 编译器 | Part 18 Triton / torch.compile / MLIR-XLA | 🚧 骨架 |
| **卷六** [vol6-inference-systems/](vol6-inference-systems/) | 推理系统 | Part 19 KV Cache / 量化 / 推理引擎 | 🚧 骨架 |
| **卷七** [vol7-models-as-operators/](vol7-models-as-operators/) | 模型的算子视角 | Part 20 Transformer / MoE 拆解 | 🚧 骨架 |

> 编号约定：Part 编号全局唯一、跨卷连续（part01~part20），卷只是分组。`src/` 下的实验目录仍按 part 编号组织，不受分卷影响。

## 为什么这样设计

市面上绝大多数 CUDA 教程回答的是 **"How"**：怎么写 kernel、怎么调用 API。
但真正决定你能不能设计出 Blackwell 上的高性能 kernel 的，是另外四个问题：

| 问题 | 说明 |
|---|---|
| **Why（为什么出现）** | 上一代架构卡在哪个硬件瓶颈上，逼着 NVIDIA 修改编程模型？ |
| **What（是什么）** | 新特性在硬件层面到底是什么电路 / 什么资源？ |
| **How（怎么编程）** | 从 CUDA C++ → PTX → SASS，软件如何驱动这个新硬件？ |
| **Evolution（如何演进）** | 下一代架构又是怎么在这个基础上继续演进的？ |

这条 **Why → What → How → Evolution** 的主线会贯穿全书每一个技术点（`cp.async`、TMA、WGMMA、TMEM、Cluster、Warp Scheduler、Scoreboard……）。只要抓住这条线，你不仅能"背下" Hopper/Blackwell 的新指令，还能**预测** Rubin、Feynman 这类未来架构大概率会往哪个方向演进——因为硬件演进本身是有内在逻辑的：**算力增长 > 访存带宽增长 > 线程/协作带宽增长**，每一代新特性几乎都是在解决"上一代被下一代算力甩开的那个环节"。

全书还有三条隐藏的主线，会在 Part 10（硬件发展史）汇合成一张全景图：

1. **算力主线**：CUDA Core → Tensor Core（Volta 起）→ 精度持续下探（FP16→INT8→TF32/BF16→FP8→FP4/FP6）→ 协作/通信范围扩展（Warp→Warp Group→跨 SM CTA Pair）。注意：数据中心 Blackwell 的部分 `tcgen05` 变体可由单线程发起；“发起粒度”与“协作范围”不是同一概念。
2. **搬运主线**：`ld.global`+`st.shared`（线程搬）→ `cp.async`（Warp 异步搬）→ TMA（专用 DMA 搬）→ TMEM 配套的 `tcgen05.cp`（结果直接落地专属存储）。
3. **协作范围主线**：Thread → Warp → Block（SM 内）→ Warp Group（Hopper）→ Thread Block Cluster / CTA Pair（跨 SM）。

## 两种读法

本书正文（Part 1~13）按**主题**组织（Memory 一章讲完所有代际的访存演进、Tensor Core 一章讲完所有代际的算力演进……），这样便于查阅和系统学习，但如果你想更直观地体会"为什么下一代架构要这样做"，建议先读：

**[`roadmap-ampere-hopper-blackwell.md`](vol1-gpu-hardware/roadmap-ampere-hopper-blackwell.md) —— 按 Ampere → Hopper → Blackwell 架构代际组织的实战路线图**，把 Part 3/4/5/6 里分散在各主题下的知识点，重新串成三条循序渐进的学习轨道（Track 1 Ampere：Tensor Core/`cp.async`/Async Pipeline；Track 2 Hopper：Warp Group/WGMMA/TMA/Distributed Shared Memory；Track 3 Blackwell：TMEM/第五代 Tensor Core/新 SM Pipeline/新 PTX-MMA 指令），每个 Track 结尾都明确写出"遗留了什么问题，逼着下一代必须继续改"，并标注了每个知识点在消费级显卡（如 RTX 50 系列）上是否可以实操验证。

## 知识体系结构（不是按 API 分类，是按因果链分类）

```
GPU 硬件发展 (Part 10, 全书灵魂)
      │  为什么会长成这样
      ▼
GPU 执行模型 (Part 1)
      │  硬件长什么样、SIMT 怎么跑
      ▼
CUDA 编程模型 (Part 2)
      │  软件如何映射到硬件
      ▼
CUDA Memory (Part 3) ── 全书最重要，占比最大
      │  数据怎么在层次化存储间流动
      ▼
执行流水线 (Part 4)
      │  指令怎么发射、怎么隐藏延迟
      ▼
Tensor Core 发展史 (Part 5) ── 全书核心
      │  算力单元怎么从 FMA 进化到 tcgen05
      ▼
同步手册 (Part 13) ── 以硬件同步原语为轴
      │  执行会合 / 内存排序 / 异步完成
      ▼
PTX (Part 6) / CUDA C++ API (Part 7) / 编译器 (Part 8) / SASS (Part 9)
      │  软件栈的四层：C++ → PTX → SASS → 机器码
      ▼
性能优化实战 (Part 11)
      │
      ▼
高性能 Kernel 源码分析 (Part 12) ── 卷二终点：读懂工业级代码
      │
      ▼
─── 单卡之外（卷三~卷七，骨架已建、内容待填充）───
      │
      ▼
互联与通信 (Part 14~15)
      │  从一颗 GPU 到一万颗 GPU：NVLink/RDMA 硬件 × 集合通信算法
      ▼
分布式训练 (Part 16~17)
      │  通信算子的使用方：DP/TP/PP/EP 与框架实现
      ▼
AI 编译器 (Part 18) ── 框架 → kernel 的中间层
      ▼
推理系统 (Part 19) ── 训练之外的另一半负载
      ▼
模型的算子视角 (Part 20) ── 给所有 kernel 一个"它服务谁"的坐标
```

## 目录（含各部分核心子主题）

### 卷一 · GPU 硬件与体系结构 —— [`vol1-gpu-hardware/`](vol1-gpu-hardware/)

### 第一部分 · GPU 体系结构基础 —— [`part01-gpu-hardware-architecture.md`](vol1-gpu-hardware/part01-gpu-hardware-architecture.md)
**核心问题：GPU 为什么长这样？为什么能快？**（本部分完全不涉及 CUDA 代码）
CPU vs GPU（Latency vs Throughput Processor）、SIMD vs SIMT、GPC/TPC/SM 物理组成、Warp Scheduler、Register File、Shared Memory/L1/L2/HBM、Grid/Block/Warp/Thread 执行模型、Divergence 与 Reconvergence、Volta 独立线程调度、Memory Hierarchy 总览、Roofline 模型（Compute Bound vs Memory Bound、Arithmetic Intensity）。

### 卷一附 · 逐型号 GPGPU 硬件手册 —— [`gpgpu-models/`](vol1-gpu-hardware/gpgpu-models/)
**核心问题：A100 / H100 / B200 每颗芯片的具体数字怎么从硬件单元推出来？**
[Ampere A100](vol1-gpu-hardware/gpgpu-models/01-ampere-a100.md)、[Hopper H100](vol1-gpu-hardware/gpgpu-models/02-hopper-h100.md)、[Blackwell B200](vol1-gpu-hardware/gpgpu-models/03-blackwell-b200.md) 三篇手册，每篇给出拓扑（GPC/TPC/SM）、SM 内部结构、算力公式与带宽公式的完整推导。另有按代际组织的实战路线图 [`roadmap-ampere-hopper-blackwell.md`](vol1-gpu-hardware/roadmap-ampere-hopper-blackwell.md)。

### 卷二 · CUDA 软件栈与 Kernel 工程 —— [`vol2-cuda-software/`](vol2-cuda-software/)

### 第二部分 · CUDA 编程模型 —— [`part02-cuda-programming-model.md`](vol2-cuda-software/part02-cuda-programming-model.md)
**核心问题：软件怎么映射到硬件？**
Grid/Block/Thread/Warp 的编程语义、**Hopper Thread Block Cluster 执行层级**（`clusterDim`/`block_rank`/共同调度；DSM 细节指向 Part 3）、Kernel Launch、Stream/Event、Occupancy、Memory Model、同步原语层级（含 `cluster.sync`）。结尾回答：为什么 Block 不能跨 SM？为什么还要 Cluster？为什么 Shared Memory 属于 Block？

### 第三部分 · CUDA Memory（全书最重要）—— [`part03-cuda-memory/00-overview.md`](vol2-cuda-software/part03-cuda-memory/00-overview.md)
**核心问题：数据怎么流动？每一种搬运机制为什么出现？**
现已扩展为 10 篇手册：Global/Shared/Register/Constant/Texture/Local Memory、Unified/Pinned Memory；Coalescing、Alignment、Bank Conflict、Swizzle；`ld.global+st.shared` → **非 bulk `cp.async`（Ampere）** → **bulk copy / TMA（Hopper）** → **Blackwell TMA 扩展**（bulk reduce、U4/U6、gather4/scatter4、im2col wide、新 swizzle、CTA pair）→ **Cluster/DSM/TMEM**；包含 TensorMap host 编码、PTX/C++ 使用、mbarrier/bulk-group 完成协议、双缓冲与性能诊断。

### 第四部分 · CUDA 执行流水线 —— [`part04-execution-pipeline.md`](vol1-gpu-hardware/part04-execution-pipeline.md)
**核心问题：指令怎么被调度、怎么隐藏延迟？**
Instruction Pipeline、Warp Scheduler 决策逻辑、Scoreboard 依赖管理、Latency Hiding、Issue/Dual Issue、Tensor Core Pipeline、Memory Pipeline/DMA、Producer-Consumer 模式、Double/Triple Buffer/Ping-Pong、**Blackwell 新 SM Pipeline（`tcgen05` 异步流水线与 CTA Pair）**、Pipeline Programming。

### 第五部分 · Tensor Core 指令全景与编程手册（全书核心）—— [`part05-tensor-core-handbook/00-overview.md`](vol2-cuda-software/part05-tensor-core-handbook/00-overview.md)
**核心问题：Tensor Core 为什么演进到 `tcgen05`，又如何从 CUDA/CuTe 验证到 PTX/SASS？**
Volta（WMMA、`mma.sync` m8n8k4、Quad Pair）→ Turing/Ampere（INT8/INT4、TF32/BF16、`mma.sync`、`ldmatrix`/swizzle）→ Hopper（Warp Group、`wgmma.mma_async`、TMA、Warp Specialization）→ Blackwell（TMEM、`tcgen05`、CTA Pair），并系统覆盖 CUTLASS/CuTe MMA Atom、GEMM 自建阶梯与 CUDA → PTX → SASS 验证；明确区分数据中心 `sm_100a`/`sm_103a` 和消费级 `sm_120`。

### 第六部分 · CUDA 特殊指令 / PTX —— [`part06-ptx-instructions.md`](vol2-cuda-software/part06-ptx-instructions.md)
**核心问题：PTX 指令对应什么硬件？**
Warp 级（`shfl.sync`/`vote.sync`/`match.sync`）、Barrier（`bar.sync`/`mbarrier`）、Memory（`ld/st.global/shared`、`cp.async`、`prefetch`）、Tensor（`mma.sync`/`ldmatrix`/`wgmma`/`tcgen05`）、Atomics（`atom`/`red`）、Special Register（`%laneid`/`%warpid`/`%smid`/`%clock64`）。

### 第七部分 · CUDA C++ API —— [`part07-cuda-cpp-api.md`](vol2-cuda-software/part07-cuda-cpp-api.md)
**核心问题：API → PTX → SASS 三层关系是什么？**
Runtime API vs Driver API、Stream/Graph、Cooperative Groups（`thread_block` / `thread_block_tile` / `grid_group` / **Hopper `cluster_group`**：`this_cluster`、`block_rank`、`cluster.sync`、`map_shared_rank`）、`cuda::pipeline`/`cuda::barrier`、`memcpy_async`、WMMA API，以及如何用 `cuobjdump`/`nvdisasm` 亲自验证 API 到机器码的映射。

### 第八部分 · CUDA 编译器 —— [`part08-cuda-compiler.md`](vol2-cuda-software/part08-cuda-compiler.md)
**核心问题：nvcc/ptxas 怎么把 CUDA 变成机器码？**
`nvcc` 整体流程、`cicc`（C++→PTX）、`ptxas`（PTX→SASS：寄存器分配/指令调度/控制码生成）、Fatbin、Cubin/ELF、JIT、Relocatable Device Code、LTO。

### 第九部分 · SASS（高级）—— [`part09-sass.md`](vol2-cuda-software/part09-sass.md)
**核心问题：怎么读机器码、分析 stall？**
Instruction Format、谓词化、控制码（Stall Count/Yield Flag/Barrier/Reuse Flag）、Register Allocation、Instruction Scheduling、为什么同样 CUDA 代码会生成不同 SASS、用 Nsight Compute 做 Warp State/Stall 归因分析。配套最小探针：[`src/part09-sass/`](../src/part09-sass/README.md)（`__fmaf_rn` → `fma.rn.f32` → `FFMA`，Blackwell/Thor 可直接跑）。

### 第十部分 · GPU 硬件发展史（全书灵魂）—— [`part10-gpu-hardware-history.md`](vol1-gpu-hardware/part10-gpu-hardware-history.md)
**核心问题：Fermi → Blackwell，每一代解决了什么问题？**
Fermi → Kepler → Maxwell → Pascal → Volta → Turing → Ampere → Hopper → Blackwell，逐代回答"增加了什么/为什么增加/软件怎么支持/性能提升在哪"，收尾给出算力/搬运/协作三条主线的统一解释框架。

### 第十一部分 · CUDA 性能优化（实战）—— [`part11-performance-optimization.md`](vol2-cuda-software/part11-performance-optimization.md)
**核心问题：拿到一个慢 kernel，该怎么系统性地优化？**
诊断优先于开药的方法论、Memory/Compute/Pipeline/Occupancy/Register 优化手段、Persistent Kernel、Kernel Fusion、Roofline+Nsight+Micro Benchmark 闭环、一份可直接使用的优化检查清单。

### 第十二部分 · 高性能 Kernel 源码分析（最终目标）—— [`part12-kernel-source-analysis.md`](vol2-cuda-software/part12-kernel-source-analysis.md)
**核心问题：真实工业级代码是怎么把前面所有知识用起来的？**
Vector Add → Reduce → Scan → Transpose → GEMM → LayerNorm → Softmax → FlashAttention → CUTLASS GEMM → FlashMLA → DeepGEMM → cuBLAS/cuDNN，逐个给出 CUDA→PTX→SASS→Hardware Pipeline 的对应关系。

### 第十三部分 · NVIDIA GPU 同步手册（Fermi → Blackwell）—— [`part13-synchronization-handbook/00-overview.md`](vol2-cuda-software/part13-synchronization-handbook/00-overview.md)
**核心问题：谁在等待、数据何时可见、异步硬件何时完成？**
不按 CUDA API 分类，而按 Hardware Synchronization Primitive 组织：CTA/Warp/Cluster/Grid execution barrier、`fence`/acquire/release/proxy fence、`cp.async`/TMA/`mbarrier`、`mma.sync`/WGMMA/`tcgen05` completion、collective 和 atomic。每项均给出 CUDA → PTX → 典型 SASS、作用域、等待与可见性语义，并明确数据中心 Blackwell 与 `sm_120` 消费级路径的差异。

### 卷三 · 互联与通信（NVIDIA 生态为主）—— [`vol3-interconnect-communication/`](vol3-interconnect-communication/)（Part 14 🚧 骨架 / Part 15 ✅ 初稿完成）

### 第十四部分 · 互联硬件 —— [`part14-interconnect-hardware/`](vol3-interconnect-communication/part14-interconnect-hardware/)
**核心问题：从一颗 GPU 到一万颗 GPU，数据走的是什么物理通路？**
[封装内 NV-HBI 双 Die](vol3-interconnect-communication/part14-interconnect-hardware/01-on-package-nv-hbi.md)、[节点内 NVLink 1~5 代 / NVSwitch / DGX-HGX 拓扑](vol3-interconnect-communication/part14-interconnect-hardware/02-node-nvlink-nvswitch.md)、[节点间 PCIe / RDMA / InfiniBand / RoCE / GPUDirect](vol3-interconnect-communication/part14-interconnect-hardware/03-cluster-rdma-network.md)、[拓扑与亲和：rail-optimized、NUMA、`nvidia-smi topo`](vol3-interconnect-communication/part14-interconnect-hardware/04-topology-mapping.md)。

### 第十五部分 · 集合通信算法与算子 —— [`part15-collective-communication/`](vol3-interconnect-communication/part15-collective-communication/)
**核心问题：AllReduce 们搬什么数据、怎么搬最快、NCCL 内部怎么实现？**
[集合通信算子语义](vol3-interconnect-communication/part15-collective-communication/01-collective-ops.md)、[Ring / Tree / Recursive HD 与带宽-延迟模型](vol3-interconnect-communication/part15-collective-communication/02-algorithms.md)、[NCCL 内部：LL/LL128/Simple 协议与拓扑感知](vol3-interconnect-communication/part15-collective-communication/03-nccl-internals.md)、[NVSHMEM 与 P2P](vol3-interconnect-communication/part15-collective-communication/04-nvshmem-p2p.md)、[通信-计算重叠与通信 Kernel 化](vol3-interconnect-communication/part15-collective-communication/05-overlap.md)。

### 卷四 · 分布式训练系统 —— [`vol4-distributed-training/`](vol4-distributed-training/) 🚧 骨架

### 第十六部分 · 并行策略 —— [`part16-parallelism/`](vol4-distributed-training/part16-parallelism/)
**核心问题：通信算子的使用方——为什么需要 AllReduce/AllToAll，由并行策略决定。**
[DP / FSDP / ZeRO](vol4-distributed-training/part16-parallelism/01-dp-fsdp-zero.md)、[TP / SP（GEMM 行列切分）](vol4-distributed-training/part16-parallelism/02-tp-sp.md)、[PP（GPipe / 1F1B / Interleaved）](vol4-distributed-training/part16-parallelism/03-pp.md)、[EP 与 MoE AllToAll](vol4-distributed-training/part16-parallelism/04-ep-moe.md)、[混合并行 → 硬件拓扑映射](vol4-distributed-training/part16-parallelism/05-hybrid-mapping.md)。

### 第十七部分 · 框架实现层 —— [`part17-framework-internals/`](vol4-distributed-training/part17-framework-internals/)
**核心问题：`torch.matmul` 这行代码到 CUDA kernel 之间发生了什么？**
[PyTorch Dispatcher / ATen](vol4-distributed-training/part17-framework-internals/01-pytorch-architecture.md)、[Autograd 引擎](vol4-distributed-training/part17-framework-internals/02-autograd.md)、[c10d 与 FSDP 源码级实现](vol4-distributed-training/part17-framework-internals/03-c10d-fsdp-impl.md)。

### 卷五 · AI 编译器 —— [`vol5-ai-compilers/`](vol5-ai-compilers/) 🚧 骨架

### 第十八部分 · AI 编译器 —— [`part18-ai-compilers/`](vol5-ai-compilers/part18-ai-compilers/)
**核心问题：框架代码如何被编译成 Part 5/6 里那些 kernel？**
[Triton：Tile 抽象到 PTX](vol5-ai-compilers/part18-ai-compilers/01-triton.md)、[torch.compile：Dynamo / AOTAutograd / Inductor](vol5-ai-compilers/part18-ai-compilers/02-torch-compile.md)、[MLIR 方言体系与 XLA](vol5-ai-compilers/part18-ai-compilers/03-mlir-xla.md)。

### 卷六 · 推理系统 —— [`vol6-inference-systems/`](vol6-inference-systems/) 🚧 骨架

### 第十九部分 · 推理系统 —— [`part19-inference/`](vol6-inference-systems/part19-inference/)
**核心问题：为什么推理是 Memory Bound + 延迟敏感，系统层面怎么应对？**
[推理 vs 训练的瓶颈模型](vol6-inference-systems/part19-inference/01-inference-vs-training.md)、[KV Cache / PagedAttention / Continuous Batching](vol6-inference-systems/part19-inference/02-kv-cache-batching.md)、[量化推理链路](vol6-inference-systems/part19-inference/03-quantization-serving.md)、[投机解码 / MoE 推理 / PD 分离](vol6-inference-systems/part19-inference/04-advanced-decoding.md)、[vLLM / SGLang / TensorRT-LLM 对比](vol6-inference-systems/part19-inference/05-engines.md)。

### 卷七 · 模型的算子视角 —— [`vol7-models-as-operators/`](vol7-models-as-operators/) 🚧 骨架

### 第二十部分 · 模型的算子拆解 —— [`part20-models/`](vol7-models-as-operators/part20-models/)
**核心问题：一个模型最终由哪些算子组成，各自吃掉多少算力、带宽和通信？**
[Transformer 的 FLOP 与显存账](vol7-models-as-operators/part20-models/01-transformer-decomposed.md)、[MoE 的 dispatch/combine 通信账](vol7-models-as-operators/part20-models/02-moe-decomposed.md)。

### 附录 A · GPGPU 术语表（通用 + NVIDIA）—— [`appendix-glossary.md`](appendix-glossary.md)
**速查用途的术语表，不是教材。** 分两部分：**GPGPU 业内通用术语**（并行计算、算力度量、存储层次、性能分析，跨厂商通用）与 **NVIDIA GPGPU 专属术语**（硬件架构 SM/GPC/TPC、执行模型 Warp/CTA/Cluster、算力单元 Tensor Core/MMA、存储互联 HBM/NVLink/TMEM、数据格式 FP64→FP4/NVFP4、软件栈 CUDA→PTX→SASS、代际特性 `cp.async`/TMA/WGMMA/`tcgen05`、工具生态 Nsight/CUTLASS/CuTe）。每个词条一句话讲清"是什么 + 为什么需要认识它"，并标注跳转到对应 Part。

## 使用建议

配套 CUDA 实验统一放在仓库根目录 `src/`，按文档 Part 和章节编号组织。当前可跑套件包括：[Part 09 SASS / FFMA 探针](../src/part09-sass/README.md)、Part 13 四维同步：[总览](../src/part13-synchronization-handbook/README.md)、[13.1 Execution Barrier](../src/part13-synchronization-handbook/01-execution-barriers/README.md)、[13.2 Memory Ordering](../src/part13-synchronization-handbook/02-memory-ordering/README.md)、[13.3 Async Pipeline](../src/part13-synchronization-handbook/03-async-pipelines/README.md)、[13.4 Tensor](../src/part13-synchronization-handbook/04-tensor-synchronization/README.md)、[13.5 Collectives/Atomics](../src/part13-synchronization-handbook/05-collectives-and-atomics/README.md)。每个 case 可生成 executable、PTX、CUBIN 与 SASS。

1. **不要跳过 Part 1**：不涉及一行 CUDA 代码，但决定了你后面看到每个 API 时是否能"看到硬件"。
2. **Part 10 建议反复读**：第一遍在读完 Part 1~2 后读一次（建立时间线），学完 Part 3~9 后再读一次（这时每个名词背后都有具体技术细节支撑）。
3. **每一个新概念，先问 Why，再看 What，再学 How，最后看 Evolution**，不要一上来就背 API 签名。
4. 建议配合真实 GPU（至少 Ampere，最好有 Hopper/Blackwell 云实例）+ Nsight Compute 实操，光看文字无法建立直觉。
5. 建议的进度节奏（对应开头 4~6 个月的估计）：
   - 第 1~2 周：Part 1~2（硬件基础 + 编程模型）
   - 第 3~6 周：Part 3~4（Memory + 流水线，全书篇幅最大的部分）
   - 第 7~9 周：Part 5（Tensor Core 演进史 + 指令编程手册，建议配合 CUTLASS 示例反复回顾）
   - 第 10~12 周：Part 6~9（PTX / C++ API / 编译器 / SASS，软件栈四层）
   - 第 13~14 周：Part 10（硬件发展史通读第二遍）
   - 第 13~14 周并行：Part 13（同步手册；在读 Async Pipeline、TMA/WGMMA 时反复查阅）
   - 第 15~18 周及以后：Part 11~12（性能优化实战 + 源码分析，建议长期持续回顾）

## 参考资料（贯穿全书）

- NVIDIA 各代架构白皮书：Fermi / Kepler / Maxwell / Pascal / Volta / Turing / Ampere / Hopper / Blackwell Whitepaper（[V100 Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)、[H100 Whitepaper](https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf)）
- [NVIDIA Hopper Architecture In-Depth（TMA/Cluster/DPX 官方技术博客）](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Inside NVIDIA Blackwell Ultra：TMEM/第五代 Tensor Core/NVFP4 官方技术博客](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [Deep Dive on the Hopper TMA Unit for FP8 GEMMs（PyTorch 官方博客）](https://pytorch.org/blog/hopper-tma-unit/)
- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on Hopper（Colfax Research）](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Controlling Data Movement to Boost Performance on Ampere（`cp.async`/`cuda::memcpy_async` 官方技术博客）](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)
- CUDA C++ Programming Guide / CUDA C++ Best Practices Guide / PTX ISA Reference Manual（各版本，随 CUDA Toolkit 发布）
- CUTLASS 官方仓库（`examples/` 教学示例 + CuTe 文档）：https://github.com/NVIDIA/cutlass
- Citadel Research：《Dissecting the NVIDIA Volta/Turing/Ampere GPU Architecture via Microbenchmarking》系列论文；以及后续针对 Blackwell 的同类微架构分析论文（如 *Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks*）
- FlashAttention / FlashAttention-2/3、FlashMLA、DeepGEMM 官方开源仓库（GitHub：Dao-AILab/flash-attention、deepseek-ai/FlashMLA、deepseek-ai/DeepGEMM）
- Nsight Compute / Nsight Systems 官方文档

## 内容概览统计

| 文件 | 大致篇幅 |
|---|---|
| `00-README.md` | 索引/总览（多卷导航） |
| `vol1-gpu-hardware/` | 卷一：Part 1 / 4 / 10 + 型号手册 + roadmap，✅ 初稿完成 |
| `vol2-cuda-software/` | 卷二：Part 2 / 3 / 5 / 6 / 7 / 8 / 9 / 11 / 12 / 13，✅ 初稿完成 |
| `vol3` ~ `vol7`（part14~20） | 卷三~卷七：互联通信 / 分布式训练 / AI 编译器 / 推理系统 / 模型算子视角；Part 15 ✅，其余 🚧 骨架 |
| `appendix-glossary.md` | 附录 A：GPGPU 术语表（通用 + NVIDIA，已含通信与分布式分组；推理分组待扩） |

如果你发现某一部分在深度、代码示例、某个具体架构细节上还想继续加深（例如某个 PTX 指令的完整语法、某个 CUTLASS 模板的逐行解读），可以直接告诉我要扩写哪一部分，我会在对应文件里继续补充，而不需要重新生成整份教程。
