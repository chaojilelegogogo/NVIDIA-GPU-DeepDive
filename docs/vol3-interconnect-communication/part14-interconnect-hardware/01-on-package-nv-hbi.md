# 14.1 封装内互联：NV-HBI 与双 Die

> 核心问题：Blackwell 的两个 die 如何像一个芯片一样工作？
>
> 前置：Part 1（SM/存储层次）、B200 手册（`vol1-gpu-hardware/gpgpu-models/03-blackwell-b200.md`）。本篇是 Part 14 的起点——我们从**最小尺度的互联**（同一封装内的两个 die）开始，逐层放大到节点内（14.2）、节点间（14.3）、集群拓扑（14.4）。

## 0. Why：为什么 Blackwell 要拆成两个 die？

从 Fermi 到 Hopper，NVIDIA 数据中心 GPU 都是**单 die（monolithic）**设计：一颗芯片上刻所有的 SM、L2、显存控制器。这个传统在 Blackwell 终止了，原因是物理制造极限：

- **光刻极限（Reticle Limit）**：光刻机单次曝光能制造的最大芯片面积约为 858 mm²。H100 的 GH100 die（814 mm²）已经顶到了天花板附近——**想继续堆算力，单 die 没有面积可涨了**。
- **良率经济学**：die 越大，晶圆上一个缺陷毁掉整颗芯片的概率越高，成本指数上升。两颗中等 die 的良率远好于一颗超大 die。

所以 B200 的答案是 **Chiplet（芯粒）**：做两颗"半大"的 die（合计 2080 亿晶体管），用先进封装拼在一起。但拆分引入了一个新问题——**两个 die 之间的数据交换走什么路？** 如果这条路慢，软件就会感知到"这其实是两颗芯片"，所有编程模型都要改。这就是 NV-HBI 要解决的问题。

## 1. What：NV-HBI 是什么

**NV-HBI（NVIDIA High-Bandwidth Interface）** 是 Blackwell 双 die 之间的片间互联：

| 指标 | 数值 | 对比 |
|---|---|---|
| 带宽 | **10 TB/s**（双向合计） | ≈ 5.5 × NVLink 5（1.8 TB/s），≈ 5 × HBM3e 显存带宽（~8 TB/s 的一半量级） |
| 连接对象 | 同一封装内的两个 compute die | 每 die 自带 4 个 HBM3e stack |
| 延迟 | 与 die 内访问同数量级 | 远低于任何出封装链路 |
| 对软件的呈现 | **一颗 GPU、一个统一地址空间** | CUDA 看不到两个 die |

关键设计目标一句话：**让 die 间互联比"出封装的任何链路"都快一个量级，快到软件可以当它不存在。**

> 类比：两颗 die 像两栋相邻的楼。NV-HBI 是楼之间的全封闭连廊——宽到你可以忽略"换楼"这件事；而 NVLink 是出园区的高速公路，PCIe 是市区道路。

## 2. How：它怎么做到"像一颗芯片"

1. **物理层**：NV-HBI 走封装内的超短距 SerDes（die-to-die 并行接口，类 CoWoS 封装上的高密度走线），距离以毫米计——距离短 → 每 bit 能耗低 → 可以把位宽做得极大（数千条并行 lane），这是 10 TB/s 的物理基础。
2. **一致性协议**：两个 die 的 L2 Cache 通过 NV-HBI 保持**硬件缓存一致**，任一 die 的 SM 访问另一 die 挂载的 HBM 时，由硬件自动路由——软件看到的是统一显存池（B200 192 GB 是一个整体，不是 2 × 96 GB）。
3. **统一调度**：GPC/TPC/SM 的编号、Block 调度、内存分配对两个 die 透明。`cudaMalloc` 不需要知道数据落在哪个 die 的 HBM 上。

> 注意与"双卡"的本质区别：两张独立 GPU 通过 NVLink 互联时，CUDA 呈现为**两个 device**，需要 P2P 显式访问（14.2）；B200 的两个 die 是**一个 device**。判据就是：die 间链路是否快到可以承接"缓存一致性流量"这种最苛刻的负载。

## 3. 同族技术：封装内互联的另外两种形态

NV-HBI 不是孤例，"封装内互联"是一个技术家族，按连接对象分：

| 技术 | 连接对象 | 带宽 | 用于 |
|---|---|---|---|
| **NV-HBI** | GPU die ↔ GPU die | 10 TB/s | B200/GB200 双 die |
| **NVLink-C2C** | GPU ↔ CPU（Grace） | 900 GB/s | GH200/GB200 超级芯片，CPU 与 GPU 共享一致内存 |
| **HBM 接口** | die ↔ 显存 stack | ~8 TB/s（B200 整卡） | 所有现代 GPU |

- **NVLink-C2C** 值得单独记住：Grace CPU 与 Hopper/Blackwell GPU 之间的封装内链路，900 GB/s 且**硬件一致**——CPU 可以直接读 GPU 显存、GPU 可以直接读 CPU 内存（LPDDR），不需要 `cudaMemcpy`。这是"统一内存"从软件模拟（Part 5 的 Unified Memory 分页迁移）走向硬件一致的关键一步。
- 行业对照：AMD 用 Infinity Fabric（MI300 的 chiplet 互联）、Intel 用 EMIB/Foveros——**chiplet + 高速 die-to-die 互联是全行业对光刻极限的共同答案**，NVIDIA 的答案是 NV-HBI。

## 4. Evolution：这条线往哪走

- **趋势 1：die 数量继续增加**。双 die 只是开始，封装技术（CoWoS 产能、混合键合）成熟后，更多 die、更细的功能拆分（计算 die / IO die / 缓存 die 分离）是大概率方向。
- **趋势 2：互联层级变多**。NV-HBI（封装内）→ NVLink（节点/机柜内）→ IB/Ethernet（集群）的三级体系已经成形；每一级带宽差一个数量级（10 TB/s → 1.8 TB/s → 0.05 TB/s/卡），**这个"每出一层壳带宽掉一个量级"的阶梯是 Part 15 通信算法分层设计（15.2 §4）和 Part 16 并行策略映射（16.5）的物理依据**。
- **对软件的含义**：封装内互联越强，"一颗 GPU"的抽象就越能往上顶；但出了封装，软件必须自己感知拓扑——这正是 14.4 的主题。

## 5. 与全书主线的呼应

- **算力主线**：单 die 面积见顶 → chiplet 是"算力继续涨"的制造侧答案，与 Tensor Core（算力电路化）、精度下探（FP8/FP4）并列为第三条算力增长路径。
- **搬运主线**：NV-HBI 是搬运层级里最新加的一层——寄存器 → SMEM → L2 → HBM → **对端 die 的 HBM** → NVLink → RDMA，Roofline 的"带宽屋顶"从此也是分层的。
- B200 手册 §1（双 Die 架构）、Part 1 §1.4（存储层次）是本篇的前传；Part 15.2（分层 AllReduce）是本篇的直接下游。

## 6. 参考资料

- [NVIDIA Blackwell 架构官方博客（双 die 与 NV-HBI）](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- B200 手册 §1（`vol1-gpu-hardware/gpgpu-models/03-blackwell-b200.md`）
- [NVIDIA NVLink-C2C 官方介绍（Grace Hopper）](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
