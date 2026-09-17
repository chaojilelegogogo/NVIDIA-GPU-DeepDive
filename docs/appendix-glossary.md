# 附录 A：GPGPU 术语表（通用 + NVIDIA）

> 定位：这是全书的**速查术语表**，不是教材。每个词条用一句话给出"是什么 + 为什么你需要认识它"，需要深入理解时跳转到对应 Part（正文里的定义和完整推导以正文为准）。
>
> 组织方式：**第一部分是 GPGPU 业内通用术语**（跨厂商、不绑定 NVIDIA），**第二部分是 NVIDIA GPGPU 专属术语**（CUDA 生态、NVIDIA 硬件/指令/软件栈）。两部分内部再按主题分组。
>
> 约定：加粗的英文是术语的**规范写法和缩写**；圆括号内是中文译名（有通行译名时使用通行译名，否则直译 + 保留英文，避免"翻译即失真"）；`code` 字体表示指令、API 或标识符。

---

## 目录

- [第一部分 · GPGPU 通用术语](#sec-1)
  - [1.1 并行计算基础](#sec-1-1)
  - [1.2 算力与性能度量](#sec-1-2)
  - [1.3 存储层次与访问模式](#sec-1-3)
  - [1.4 性能分析与优化](#sec-1-4)
- [第二部分 · NVIDIA GPGPU 专属术语](#sec-2)
  - [2.1 硬件架构](#sec-2-1)
  - [2.2 执行模型](#sec-2-2)
  - [2.3 算力单元](#sec-2-3)
  - [2.4 存储与互联](#sec-2-4)
  - [2.5 数据格式与精度](#sec-2-5)
  - [2.6 软件栈（CUDA C++ → PTX → SASS）](#sec-2-6)
  - [2.7 代际新特性（Ampere → Hopper → Blackwell）](#sec-2-7)
  - [2.8 工具与生态](#sec-2-8)

---

<a id="sec-1"></a>

# 第一部分 · GPGPU 通用术语

<a id="sec-1-1"></a>

## 1.1 并行计算基础

- **GPGPU**（General-Purpose computing on Graphics Processing Units，通用 GPU 计算）：把原本用于图形渲染的 GPU 用来做通用计算（科学计算、AI、物理仿真等）。GPGPU 不是某一种具体硬件，而是"用 GPU 跑非图形负载"这件事的统称。
- **GPU**（Graphics Processing Unit，图形处理器）：本文语境下，GPU 已经从"图形卡"演变为"并行吞吐处理器"，CUDA 把它抽象成一个能执行海量线程的通用计算设备。见 Part 1 §1.1。
- **CPU**（Central Processing Unit，中央处理器）：延迟优化型处理器，与 GPU 相对。见 Part 1 §1.1。
- **Throughput Processor**（吞吐量处理器）：以"单位时间完成的总工作量"为目标的设计哲学，GPU 属于此类。相对的是 **Latency Processor**（延迟处理器，CPU 属于此类）。见 Part 1 §1.1.1。
- **Latency Hiding**（延迟隐藏）：GPU 不降低单次访存延迟，而是靠"切换执行其它就绪线程"把等待时间掩盖掉。这是 GPU 需要海量线程的根本原因。见 Part 1 §1.1.2。
- **Data Parallelism**（数据并行）：同一段代码作用在大量独立数据上的并行模式，是 GPU 适合加速的问题的共性。见 Part 1 §1.1.2。
- **Task Parallelism**（任务并行）：不同任务在不同核上并行执行，与数据并行相对。GPU 对跨任务并行（尤其强依赖、分支密集的任务）收益有限。
- **ALU**（Arithmetic Logic Unit，算术逻辑单元）：执行加减乘除、逻辑运算的电路，是标量计算的最小单元。GPU 的算力主要由海量 ALU 堆出来。
- **FPU**（Floating-Point Unit，浮点运算单元）：专门处理浮点数的 ALU。参见 [FMA](#sec-1-2) 与 CUDA Core。
- **Core**（核心）：语义随语境变化的词。GPU 里"core"通常指单个标量 ALU/FPU（如 CUDA Core），而 CPU 的"core"指包含取指/译码/乱序等完整逻辑的独立处理器。

<a id="sec-1-2"></a>

## 1.2 算力与性能度量

- **FLOP**（FLoating-point OPeration，浮点运算）：一次浮点运算，是算力的基本计数单位。注意大小写：FLOP 是"操作次数"，FLOPS 是"每秒操作次数"。
- **FLOPS**（FLoating-point Operations Per Second，每秒浮点运算次数）：算力的速率单位，读作"flops"。常见数量级前缀：GFLOPS（1e9）、**TFLOPS**（1e12）、**PFLOPS**（1e15）、**EFLOPS**（1e18）。本文常用 TFLOPS（如 A100 FP16 稠密 312 TFLOPS）。
- **OPS / TOPS**（Operations Per Second / Tera-OPS，每秒运算次数 / 每秒万亿次运算）：整数运算的算力单位。因为整数不分"乘/加"的浮点语义，业界常用 TOPS 而非 TFLOPS（如 A100 INT8 624 TOPS）。主要见于 AI 推理芯片的规格宣传。
- **FMA**（Fused Multiply-Add，融合乘加）：一条指令完成 `a×b+c`，只做一次舍入。GPU 的标量算力几乎全部由 FMA 单元提供。现代 GPU 一个 core 每时钟 1 个 FMA。见 Part 1。
- **MAC**（Multiply-Accumulate，乘累加）：`累加器 += a×b` 的运算，在矩阵乘/Tensor Core 语境下是基本操作。**每个 MAC = 1 次乘 + 1 次加 = 2 FLOP**，这是所有算力公式里"×2"的根源。见 A100 手册 §3.1。
- **FLOP/Byte**（每字节浮点运算次数）：**算术强度（Arithmetic Intensity）**的单位，见 1.4 的 Roofline。
- **FLOPS/Watt**（每瓦算力/能效比）：算力除以功耗，衡量能效。数据中心 GPU 的能效是重要设计约束（解释了很多"为什么不用更高频率"的取舍）。

<a id="sec-1-3"></a>

## 1.3 存储层次与访问模式

- **Memory Hierarchy**（存储层次/内存层级）：从快到慢、从贵到便宜、从近到远的多级存储（寄存器 → 缓存/Local → 主存/显存 → 持久存储）。分层设计是为了在"速度"与"容量/成本"之间折中。见 Part 1 §1.4、Part 3。
- **Bandwidth**（带宽）：单位时间能搬多少数据（GB/s、TB/s）。GPU 的访存吞吐由带宽决定，是 Roofline 的另一半。见 Part 1 §1.6。
- **Latency**（延迟）：从发出访问到数据可用的时钟周期数。GPU 通过延迟隐藏应对高延迟，而非降低延迟。见 Part 1 §1.1.2。
- **Cache**（缓存）：硬件自动管理的高速片上存储。GPU 的 Cache 主要目的是**节流**（减少对带宽的依赖）而非**降延迟**，与 CPU 的定位不同。见 Part 1 §1.4.1。
- **Cache Line**（缓存行）：缓存以固定大小（常为 32 B、64 B、128 B）为单位读写的最小粒度。访存是否高效，很大程度取决于能否"一整行都被用上"。
- **Coalesced Access**（合并访问/合并访存）：一个 Warp 内 32 个线程访问的内存地址落在同一（或少数几个）缓存行内，可被合并成一次宽访问。合并是 GPU 访存效率的第一准则。见 Part 3 §2。
- **Alignment**（对齐）：数据地址对齐到其大小（以及缓存行）的边界，是合并访问的前提之一。
- **Bank Conflict**（Bank 冲突）：Shared Memory 被划分为多个 Bank，同一时钟内多个线程访问不同地址但落到同一 Bank 时，访问被串行化。见 Part 3 §2。
- **Swizzle**（乱序/交织）：对共享内存中数据的存放顺序做重排，让矩阵 tile 的访问模式能避开 Bank 冲突。Tensor Core 的 `ldmatrix` 输入常依赖特定 swizzle 布局。见 Part 5。
- **Locality**（局部性）：时间局部性（最近用过的数据很快再用）与空间局部性（相邻数据很快被访问）。缓存、合并访存、Shared Memory 复用都建立在这个概念上。

<a id="sec-1-4"></a>

## 1.4 性能分析与优化

- **Roofline Model**（屋顶线模型）：用一张图同时表示算力峰值与带宽上限，判断某 kernel 是"算力受限"还是"访存受限"，以及优化应该往哪个方向。见 Part 1 §1.6。
- **Arithmetic Intensity**（算术强度）：`FLOP / Byte`，一个 kernel 每搬 1 字节数据能做的浮点运算数。这是 Roofline 模型的横轴。见 Part 1 §1.6。
- **Ridge Point**（脊点）：Roofline 图中算力屋顶线与带宽屋顶线的交点，对应"临界算术强度"。算术强度低于它的 kernel 是访存受限，高于它是算力受限。见 Part 1 §1.6。
- **Compute Bound**（算力受限）：kernel 的瓶颈是算力（ALU/Tensor Core 忙不过来），优化重点是提高计算单元利用率。见 Part 1 §1.6。
- **Memory Bound / Bandwidth Bound**（访存受限/带宽受限）：kernel 的瓶颈是带宽（等数据搬不过来），优化重点是减少访存、提高访存效率。见 Part 1 §1.6。
- **Latency Bound**（延迟受限）：kernel 的瓶颈是延迟（没有足够并行度来隐藏延迟），常表现为 [Occupancy](#sec-2-2) 不足。见 Part 11。
- **Stall**（停顿/停滞）：执行单元因等待（数据未就绪、资源冲突等）而无法发射指令。GPU 靠切换其它 Warp 掩盖 stall。见 Part 4、Part 9。
- **Benchmark / Microbenchmark**（基准测试/微基准）：针对单一硬件行为（如某指令吞吐、某访存模式延迟）编写的小型测量程序，用于反推微架构细节。本教程大量结论来自 Citadel Research 等微基准论文。
- **Profiling**（性能剖析）：用工具（Nsight Compute/Systems、`nvprof` 等）采集 kernel 运行时的硬件计数器，定位瓶颈。见 Part 9、Part 11。

---

<a id="sec-2"></a>

# 第二部分 · NVIDIA GPGPU 专属术语

<a id="sec-2-1"></a>

## 2.1 硬件架构

- **SM**（Streaming Multiprocessor，流式多处理器）：NVIDIA GPU 的核心计算单元，是**软件能感知到的最小硬件单元**。一个 SM 内含若干 Warp Scheduler、一组 CUDA Core / Tensor Core、寄存器堆、Shared Memory、L1/Cache、若干专用单元（LD/ST、SFU、TMA）。Block 被调度到一个 SM 内执行。见 Part 1 §1.2。
- **GPC**（Graphics Processing Cluster，图形处理簇）：SM 的高层分组单元。一个 GPC 内含多个 TPC/SM，GPC 内部共享部分资源（如 Raster 单元、L2 分片入口）。见 Part 1 §1.2。
- **TPC**（Texture Processing Cluster，纹理处理簇）：GPC 与 SM 之间的中间层级，通常含 1~2 个 SM + 纹理相关单元。在数据中心 GPGPU 语境下其"纹理"职责已弱化，更多作为物理组织层级存在。见 Part 1 §1.2。
- **CUDA Core**：NVIDIA 对"标量浮点/整数 ALU"的市场命名，1 个 CUDA Core ≈ 1 条 FMA/时钟的标量算力。注意：不同架构下 FP32 与 INT32 是否共用同一 core 是有差异的（Ampere 消费级 GA10x 是"混用 128 通路二选一"，数据中心 GA100 是"64 FP32 + 64 INT32 分立"）。见 A100 手册 §1.1。
- **Tensor Core**：NVIDIA 专用于矩阵乘累加（MMA）的硬件单元，是 Volta 之后 AI 算力的核心载体。按"每时钟 MAC 数"衡量，随代际翻倍（第 3 代 A100 → 第 4 代 H100 → 第 5 代 B200）。见 Part 5、A100/H100/B200 手册。
- **SFU**（Special Function Unit，特殊函数单元）：处理超越函数（`sin`、`exp`、`log`、开方、倒数等）的专用单元，吞吐远低于普通 CUDA Core。见 Part 1。
- **LD/ST Unit**（Load/Store Unit，访存单元）：SM 内负责读/写显存与 Shared Memory 的专用单元，与计算单元相对独立。TMA/DMA 的出现就是为了进一步从普通访存路径中分离出"搬运"职责。见 Part 3、Part 4。
- **Die**（裸片/晶粒）：一颗芯片（从晶圆上切割下来的那个物理芯片）。B200 是**双 Die（chiplet）**设计，两个 die 通过 NV-HBI 互联封装成一颗。见 B200 手册。
- **Chiplet**（小芯片/芯粒）：把一颗大芯片拆成多个小 die 封装到一起的制造思路，用于突破单一 die 的曝光极限（reticle limit）。NVIDIA 在 Blackwell 数据中心线首次大规模采用。见 B200 手册 §1。
- **Transistor**（晶体管）：芯片的基本单元，数量（如 H100 800 亿、B200 2080 亿）是衡量芯片规模与工艺的常用指标。

<a id="sec-2-2"></a>

## 2.2 执行模型

- **CUDA**（Compute Unified Device Architecture）：NVIDIA 的 GPGPU 编程模型 + 软件栈的总称（既是编程模型的名称，也是一套 C/C++ 扩展与库的统称）。
- **Kernel**（内核/核函数）：在 GPU（Device）上执行的、由 Host 调用的函数。`__global__` 声明的就是 kernel。一次 kernel 启动会生成海量线程。见 Part 2。
- **Host / Device**（主机/设备）：Host 指 CPU 及其内存，Device 指 GPU 及其显存。CUDA 编程围绕"Host 发起、Device 执行、显式拷贝"展开。
- **Thread**（线程）：GPU 执行的最小单元，对应一条标量指令流。逻辑上每个线程独立，物理上被捆成 Warp 锁步执行。见 Part 1 §1.1.4、Part 2。
- **Warp**（线程束，常不翻译）：32 个线程组成的基础执行单位。硬件以 Warp 为粒度发射指令、以锁步（lockstep）方式驱动这 32 个线程。**为什么是 32** 是 Part 1 §1.3 讨论的关键问题（指令发射效率与分支代价的权衡）。见 Part 1、Part 2。
- **Warp Scheduler**（Warp 调度器）：SM 内负责挑选"下一发发射哪个 Warp 的哪条指令"的硬件单元。每个 SM 有多个 Warp Scheduler，决定并发度与调度行为。见 Part 1 §1.2、Part 4。
- **Lane**（通道）：Warp 内 32 个线程各自对应的物理执行通道。`laneid` 是 0~31 的线程在束内编号。
- **Divergence**（分歧）：Warp 内线程走到不同分支路径时，硬件串行执行各分支，导致性能下降。见 Part 1 §1.4。
- **Reconvergence**（重汇聚）：分歧的分支汇合后线程重新合并为同步执行。现代 Volta+ 支持独立线程调度（Independent Thread Scheduling），重汇聚可延迟。见 Part 1 §1.4。
- **Grid / Block / Thread**（网格/线程块/线程）：CUDA 的三层线程组织。Block 包含多个 Thread，Grid 包含多个 Block。Block 被调度到单一 SM 内（不能跨 SM），是"可共享 Shared Memory + 可做块内同步"的边界。见 Part 2。
- **BlockIdx / ThreadIdx**：CUDA 内建变量，用于标识线程在 Grid/Block 中的位置，是"每个线程处理哪份数据"的寻址依据。见 Part 2。
- **Occupancy**（占用率）：SM 上"实际驻留的活跃 Warp 数 / 硬件理论最大 Warp 数"的比值（如 A100 每 SM 最多 64 Warp，驻留 32 个即 50% 占用率）。分**理论占用率**（Theoretical Occupancy，由 kernel 资源用量算出）与**实际占用率**（Achieved Occupancy，运行时实测，受调度与负载均衡影响）。占用率被三个资源上限共同钳制：**每线程寄存器数 × 线程数 ≤ 寄存器堆容量**、**每 Block Shared Memory × Block 数 ≤ SM 共享内存容量**、**Block 数 ≤ SM 最大 Block 槽位数**（另有每 SM 线程数上限），任一打满都会截断驻留 Warp 数。高占用率 = 更多可切换的就绪 Warp = 更强的延迟隐藏能力（直接决定能否摆脱 [Latency Bound](#sec-1-4)）；但占用率不是唯一目标——寄存器重度使用的 kernel 往往以较低占用率换取每线程更多 ILP，同样能打满算力。可用 `--ptxas-options=-v` 查看资源用量、用 CUDA Occupancy Calculator 或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 估算。见 Part 2、Part 11。
- **Resident Threads / Resident Warps**（驻留线程/驻留 Warp）：当前同时驻留在 SM 上、可被调度的线程/Warp 总数，是延迟隐藏能力的直接来源。见 Part 4。
- **Thread Block Cluster / Cluster**（线程块簇）：Hopper 引入的执行层级，把多个 Block 绑定为一簇、可在同一组 SM 上共同调度并跨 SM 访问 Shared Memory（配合 DSM）。见 Part 2、Part 3 §10。
- **Warp Group**（Warp 组）：4 个连续 Warp（128 线程）的协作颗粒度，Hopper 的 WGMMA 以它为发起单位。见 Part 5、H100 手册。
- **CTA**（Cooperative Thread Array，协作线程数组）：**Thread Block 的技术术语**，二者等价，常见于 PTX 与白皮书语境（如 `cp.async.bulk.*.shared::cta`、CTA Pair）。见 Part 2。
- **CTA Pair**（CTA 对/协作线程数组对）：Blackwell 的新 SM 流水线中，同一 TPC 内相邻两个 SM 组成一对，共享一次 TMA 搬运的输入喂给两份 Tensor Core。见 B200 手册 §3.3、Part 4 §4.10。

<a id="sec-2-3"></a>

## 2.3 算力单元

- **MMA**（Matrix Multiply-Accumulate，矩阵乘累加）：Tensor Core 的核心操作 `D = A×B + C`，一条指令完成一个 tile 的矩阵乘累加。见 Part 5。
- **FMA 单元 vs Tensor Core**：标量算力（CUDA Core，逐个标量 FMA）vs 矩阵算力（Tensor Core，一次 tile MMA）的两条算力通路。见 Part 1、Part 5。
- **`mma.sync`**：Warp 级协作的 Tensor Core 指令（Ampere 起），要求 32 线程协同发出一条 MMA 指令。原子形状如 `m16n8k8` / `m16n8k16`。见 Part 5 §4。
- **`wgmma.mma_async`**：Hopper 的 Warp Group 级异步 Tensor Core 指令，操作数可直接从 Shared Memory 读取（配合矩阵描述符），tile 比 `mma.sync` 大得多。见 Part 5 §5、H100 手册。
- **`tcgen05`**：Blackwell 数据中心线的 Tensor Core 指令族（`tcgen05.mma` / `ld` / `st` / `cp` / `alloc` / `dealloc` 等），结果直接写入 TMEM，可由单线程发起。见 Part 5 §6、B200 手册。
- **`wmma`**：Volta/Turing 时代的高层 WMMA API（`nvcuda::wmma`），是 Tensor Core 最早的编程接口，Ampere 起被更底层的 `mma.sync` 与 `ldmatrix` 取代。见 Part 5 §3。
- **`ldmatrix`**：把 Shared Memory 中的矩阵 tile 高效搬到寄存器的工作区，配合 `mma.sync` 使用，是 Ampere 时代喂 Tensor Core 的关键指令。见 Part 5 §4。
- **Sparsity（2:4 Structured Sparsity，2:4 结构化稀疏）**：权重每 4 个元素里至少 2 个为 0，硬件直接跳过零从而峰值翻倍。Ampere 引入。见 A100 手册 §3。

<a id="sec-2-4"></a>

## 2.4 存储与互联

- **Register File**（寄存器堆）：SM 内每个线程独占的一份寄存器存储，大小如 64K × 32-bit（256 KB/SM）。它是速度最快、延迟最低的一级存储。寄存器堆容量"十年不涨"是 Hopper/WGMMA、Blackwell/TMEM 出现的深层原因。见 Part 1 §1.4、A100/H100/B200 手册。
- **Register Spilling**（寄存器溢出/寄存器溢出到 local）：寄存器不够用时，编译器把变量溢出到 Local Memory（其实是显存），导致性能骤降。见 Part 2、Part 11。
- **Shared Memory（SMEM）**：SM 内、Block 内所有线程共享的软件管理片上存储，比 Global 快一个数量级，是可编程性最强的缓存。见 Part 3 §1。
- **L1 Cache / L2 Cache**：L1 在 SM 内（常与 Shared Memory 共享同一片 SRAM，可配置划分比例），L2 在整颗 GPU 上被所有 SM 共享（A100 40 MB → H100 50 MB → B200 126 MB）。见 Part 1 §1.4、Part 3 §1。
- **Local Memory**：逻辑上"线程私有"，物理上落在显存（部分经 L1/L2 缓存），用于放溢出寄存器与大数组。见 Part 3 §1。
- **Global Memory**：显存主区域，容量最大、延迟最高，所有线程可访问。kernel 的主要输入输出都从这里读入/写出。见 Part 3 §1。
- **Constant Memory / Texture Memory**：只读且优化的特殊存储（常量缓存、纹理缓存），适合广播式只读访问。见 Part 3 §1。
- **Unified Memory / Pinned Memory**：统一内存（`cudaMallocManaged`）让 CPU/GPU 共享同一地址空间；固定内存（`cudaMallocHost`）是页锁定的 Host 内存，支持更快的 DMA 传输。见 Part 3 §1。
- **HBM**（High Bandwidth Memory，高带宽内存）：GPU 用的高带宽显存堆叠（HBM2/HBM2e/HBM3/HBM3e），通过超宽总线（如 5120-bit）获得 TB/s 级带宽。见 Part 1 §1.4、各型号手册。
- **NVLink**：NVIDIA 的多 GPU 高速互联（card-to-card），用于多卡间点对点高带宽通信（如 DGX 机箱内 8 卡全互联）。A100 为 Gen3 600 GB/s。见 A100 手册 §5。
- **NVSwitch**：连接多个 GPU 的交换机芯片，让多卡间实现全带宽直连。
- **NV-HBI**（NVIDIA High-Bandwidth Interface）：Blackwell 双 Die 之间的片间互联，提供 10 TB/s 级带内双向带宽，让两个 die 像一个芯片一样工作。见 B200 手册 §1。
- **NVHS**（NVLink High Speed）：Blackwell 一代的 NVLink 高速互联（第五代，1800 GB/s/GPU）。见 B200 手册。

<a id="sec-2-5"></a>

## 2.5 数据格式与精度

> 说明：所有"数据类型 → 吞吐"的递变关系，底层逻辑都是同一套 Tensor Core 电路，输入位宽越窄、每时钟塞进去的 MAC 越多。见 A100 手册 §3.2。

- **FP64**（Double Precision，双精度浮点）：64 位，科学计算/高精度仿真主力。GPU 上通常被严格限制速率（A100 9.7、H100 33.5、B200 40 TFLOPS），是数据中心芯片与消费卡的关键分水岭。见各型号手册 §4。
- **FP32**（Single Precision，单精度浮点）：32 位，通用计算默认精度。
- **TF32**（TensorFloat-32）：19 位输入（FP32 的 8 位指数 + 10 位尾数）的"加速深度学习"格式，Ampere 引入，用于把 FP32 输入喂进 Tensor Core 时获得 FP16 级吞吐、接近 FP32 的数值行为。见 A100 手册。
- **FP16**（Half Precision，半精度浮点）：16 位，AI 训练/推理主流格式（H100 FP16 稠密 989 TFLOPS）。
- **BF16**（Brain Floating Point 16）：16 位，指数 8 位 + 尾数 7 位（范围对齐 FP32、精度低于 FP16），更抗溢出，多用于训练。与 FP16 在硬件上同吞吐。见 A100 手册。
- **FP8**（8 位浮点）：Hopper 引入，分 E4M3（4 指数 3 尾数，精度高）与 E5M2（5 指数 2 尾数，范围大）两种，用于大模型推理/训练的前向。见 H100 手册。
- **FP6**（6 位浮点）：Blackwell 新增，分 E3M2 / E2M3，吞吐介于 FP8 与 FP4 之间。见 B200 手册 §4.2。
- **FP4**（4 位浮点）：Blackwell 引入的极致低精度（E2M1 等），峰值可达 9000 TFLOPS（B200 稠密）。见 B200 手册。
- **NVFP4**（NVIDIA FP4 with Block Scale，带块缩放的 FP4）：Blackwell 的两级缩放机制——16 个 FP4 值共享一个 FP8 微块缩放因子，外加张量级 FP32 缩放，用于把 FP4 的量化误差压到可用范围。见 B200 手册 §5。
- **INT8 / INT4**（8 位/4 位整数）：整数低精度格式，用于推理量化（A100 INT8 624 TOPS、INT4 1248 TOPS）。
- **Quantization**（量化）：把高精度权重/激活映射到低精度整数或低精度浮点，以换吞吐与带宽，代价是精度损失。缩放因子（scale）是量化恢复差异的关键。见 Part 5、B200 手册。
- **Scale / Block Scale**（缩放/块缩放）：量化里的缩放因子；（块缩放）把一组元素共用一个缩放因子，是 FP4/FP8 可用化的关键设计。

<a id="sec-2-6"></a>

## 2.6 软件栈（CUDA C++ → PTX → SASS）

- **NVCC**：NVIDIA CUDA 编译器驱动，把 `.cu` 编译为 Device 代码 + Host 代码的可执行程序。见 Part 8。
- **CICC**：CUDA 编译器前端（`cicc`），负责 C++ → PTX 的编译阶段。见 Part 8。
- **PTXAS**（PTX Assembler）：把 PTX 汇编为 SASS 的汇编器，负责寄存器分配、指令调度、控制码生成等。见 Part 8。
- **PTX**（Parallel Thread Execution，并行线程执行）：NVIDIA 的虚拟 ISA（中间表示），跨代兼容（新硬件能用 JIT 兼容旧 PTX）。是 CUDA C++ 与 SASS 之间的中间层。见 Part 6、Part 8。
- **SASS**（Shader Assembly）：PTX 经 `ptxas` 汇编后的**真实机器码**（每代架构各不相同），用 `cuobjdump -sass` 或 `nvdisasm` 查看。见 Part 9。
- **CUBIN**（CUDA Binary）：编译/汇编产物，含 SASS 的 ELF 二进制。见 Part 8。
- **Fatbin**（Fat Binary）：在一个可执行文件里同时打包多个架构（多个 sm_XX）版本的 CUBIN，运行时按实际 GPU 选择。见 Part 8。
- **JIT**（Just-In-Time Compilation，即时编译）：PTX 在运行时被驱动即时编译成当前硬件的 SASS，是"老 PTX 跑新硬件"的机制。见 Part 8。
- **Relocatable Device Code（RDC）**：可重定位设备代码（`-rdc=true`），允许跨编译单元链接 Device 函数。见 Part 8。
- **`sm_XX` / Compute Capability**：NVIDIA 的计算能力编号，标识架构代际与指令能力（如 `sm_80` Ampere A100、`sm_86` Ampere 消费级、`sm_90a` Hopper、`sm_100a` 数据中心 Blackwell、`sm_120` 消费级 Blackwell）。`a` 后缀表示"架构专属特性"（如 wgmma/tcgen05）。编译时用 `-arch=sm_XX` 指定。见 Part 8、roadmap。

<a id="sec-2-7"></a>

## 2.7 代际新特性（Ampere → Hopper → Blackwell）

> 这些是本书"Why → What → How → Evolution"主线落地的核心词。详见 roadmap 与各型号手册。

- **`cp.async`**（Asynchronous Copy，异步拷贝，Ampere）：把 Global→Shared 的搬运改成硬件异步执行、不经过发起线程的寄存器，是"喂饱 Tensor Core"的第一步。见 Part 3 §4、A100 手册。
- **Async Pipeline / Double Buffer**（异步流水线/双缓冲）：让"搬运下一批数据"与"计算当前批"重叠的编程范式，`cp.async` 的收益要靠它才能兑现。见 Part 4 §4.8~4.9。
- **TMA**（Tensor Memory Accelerator，张量内存加速器，Hopper）：SM 内的专用 DMA 引擎，单线程发一条描述符指令即可完成规整张量块的搬移，彻底把"搬运"从 Warp 的指令发射里剥离。见 Part 3 §5、H100 手册。
- **TensorMap**（张量映射描述符）：TMA 搬运用的描述符，在 Host 端用 `cuTensorMapEncodeTiled` 编码 tile 维度/边界/步长信息。见 Part 3 §5。
- **`mbarrier`**（Memory Barrier / Async Barrier，异步屏障）：Hopper 的异步事务屏障，配合 TMA/`cp.async.bulk` 做完成的到达/等待，支持 expect_tx（预期事务数）。见 Part 3、Part 13。
- **WGMMA**：见 2.3 `wgmma.mma_async`。
- **TMA Multicast**：TMA 的一次搬运广播到多个 Block（Cluster 内），依赖 DSM 的地址空间。见 Part 3 §5。
- **DSM**（Distributed Shared Memory，分布式共享内存，Hopper）：让 Cluster 内的 Block 跨 SM 直接访问对方的 Shared Memory（`cluster.map_shared_rank`、`.shared::cluster`、`mapa`）。见 Part 3 §10、H100 手册。
- **Warp Specialization**（Warp 专职化）：把生产者（搬运）Warp 组与消费者（计算）Warp 组分开专职各司其职的编程范式，Hopper 上用于最大化 Tensor Core 利用率。见 Part 4 §4.8、H100 手册。
- **TMEM**（Tensor Memory，张量内存，Blackwell）：数据中心 Blackwell 给 Tensor Core 配的专属累加器存储（256 KB/SM），把越来越大的 MMA 累加器从寄存器堆搬出来，是 `tcgen05` 结果落地的目标。见 B200 手册 §2.2、Part 3 §11。
- **`tcgen05`**：见 2.3。
- **CTA Pair**：见 2.2。
- **Transformer Engine**（Transformer 引擎）：Hopper/Blackwell 上的硬件+软件机制，用于在 FP8/FP4 等低精度下做动态精度管理（按层/按张量选择精度与缩放）。Hopper 第一代、Blackwell 第二代。见 H100/B200 手册。
- **DPX**（Dynamic Programming X-Instructions，动态规划指令）：Hopper 引入的专用指令族（用于路径规划、序列比对等动态规划问题），体现"领域专用加速"方向。见 H100 手册。

<a id="sec-2-8"></a>

## 2.8 工具与生态

- **Nsight Compute（NCU）**：NVIDIA 的 kernel 级性能剖析器，可看 Warp State、Stall 归因、驻留 Warp、访存模式等硬件计数器，是性能优化的核心工具。见 Part 9、Part 11。
- **Nsight Systems（NSYS）**：系统级/时间线剖析器，适合看 Host/Device 交互、拷贝与计算的 overlap、kernel 启动开销。见 Part 11。
- **`cuobjdump` / `nvdisasm`**：查看 CUBIN/SASS 的命令行工具（`cuobjdump -sass` 反汇编出 SASS）。是"验证 CUDA → PTX → SASS 映射"的必备工具。见 Part 6、Part 8、Part 9。
- **CUDA Runtime API / Driver API**：Runtime API（`cudaMalloc`、`cudaMemcpy`、`<<<>>>` 启动）在 Driver API 之上封装了自动化；Driver API（`cu*`）更底层、可显式管理 Context。见 Part 7。
- **Stream / Event / Graph**：Stream 是异步操作的执行队列（不同 Stream 可并发/overlap），Event 用于流间同步与计时，Graph（CUDA Graph）把一批 kernel 预录制为图以降低启动开销。见 Part 7。
- **Cooperative Groups**：CUDA 的线程协作抽象库（`thread_block`、`thread_block_tile`、`grid_group`、Hopper 的 `cluster_group`）。见 Part 7。
- **CUTLASS**：NVIDIA 开源的通用矩阵乘模板库，是学习高性能 Tensor Core kernel 的事实标准（`examples/` + CuTe 教学）。见 Part 5、Part 12。
- **CuTe**：CUTLASS 3.x 的底层抽象（Layout/Tensor/Copy/MMA Atom），是理解 WGMMA/tcgen05 编程的关键入口。见 Part 5。
- **cuBLAS / cuDNN / cuSPARSE**：NVIDIA 官方的线性代数库 / 深度神经网络库 / 稀疏矩阵库，工业级实现可作性能参照。见 Part 12。
- **NVCC Flags：`-arch` / `-use_fast_math` / `--ptxas-options=-v`**：`-arch=sm_XX` 指定目标架构；`-use_fast_math` 启用快速数学（牺牲精度换吞吐）；`--ptxas-options=-v` 打印寄存器/Spill 用量统计。见 Part 8、Part 11。
- **FlashAttention / FlashMLA / DeepGEMM**：社区开源的高性能 Attention / GEMM kernel 代表作，是 Part 12 源码分析的最终目标。见 Part 12。

---

### 关于"为什么这两个词没单独列"

- **SIMT**（Single Instruction Multiple Threads）：虽常被视为 NVIDIA 专属，但它本质是"披着多线程外衣的 SIMD"这一通用思想，故放在第一部分 §1.1 相关处讨论（Part 1 §1.1.4 有完整推导）。
- **SIMD**（Single Instruction Multiple Data）：属于并行计算基座概念，见 Part 1 §1.1.3。