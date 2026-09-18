# 14.3 节点间互联：PCIe、RDMA、InfiniBand、RoCE 与 GPUDirect

> 核心问题：跨机器的 GPU 如何绕过 CPU 直接读写对方显存？
>
> 前置：14.2（节点内 NVLink）。出了机箱，带宽从 900 GB/s 掉到 50 GB/s/卡——本篇讲这"最后一层"互联的技术栈，以及为什么它的每个设计都在围绕一件事：**让数据不碰 CPU、不碰主存**。

## 0. 先看全局：三级互联的带宽阶梯

把 14.1~14.3 放在一起，一次 GPU 间通信能走的路是一个带宽阶梯：

| 层级 | 技术 | 带宽（每 GPU） | 延迟量级 |
|---|---|---|---|
| 封装内 | NV-HBI（B200 双 die） | 10 TB/s | ~百 ns |
| 节点/机柜内 | NVLink 5 + NVSwitch | 1.8 TB/s | ~µs |
| 节点间 | IB NDR 400G 网卡 | 50 GB/s | ~数 µs |

**每出一层壳，带宽掉约一个数量级。** 这个阶梯是分布式系统设计的物理宪法：15.2 的分层 AllReduce、16.5 的"TP 放节点内、DP 放节点间"，全是在向这张表妥协。

## 1. PCIe：节点间的"地板"，也是网卡的入口

跨节点通信的物理出口是网卡（NIC），而网卡插在 PCIe 上——所以先理解 PCIe：

| PCIe 代 | 单 lane 速率 | x16 双向合计 | 搭载平台 |
|---|---|---|---|
| Gen3 | 1 GB/s | ≈ 32 GB/s | Volta 及以前 |
| Gen4 | 2 GB/s | ≈ 64 GB/s | A100 |
| Gen5 | 4 GB/s | ≈ 128 GB/s | H100/B200 |
| Gen6 | 8 GB/s（PAM4） | ≈ 256 GB/s | Rubin 世代 |

关键认知：**PCIe 是"主机总线"，为 CPU 为中心的世界设计**——设备间的流量默认要经过 CPU 根复合体（Root Complex）仲裁。一张 400G 网卡需要 50 GB/s，PCIe Gen5 x16 单向 64 GB/s 刚刚够——**网卡已经把 PCIe 吃满了**，这就是为什么 GPU 间通信不能再容忍"绕道主存"。

## 2. RDMA：把 CPU 从数据面上踢出去

### 2.1 传统 TCP 路径的问题

用普通 TCP socket 把 GPU 数据发到另一台机器，数据路径是：

```
GPU 显存 →（拷贝）→ 主存用户态 buffer →（拷贝）→ 内核 socket buffer
→（协议栈处理，CPU 逐包参与）→ 网卡 → 网络 → 对端逆向重演一遍
```

4 次拷贝 + CPU 全程参与 + 内核上下文切换——延迟几十 µs 起步，CPU 占用高，且带宽被拷贝路径拖累。

### 2.2 RDMA 的三个支柱

**RDMA（Remote Direct Memory Access，远程直接内存访问）** 的设计哲学是把这三样全部消灭：

1. **内核旁路（Kernel Bypass）**：应用程序直接把收发请求交给网卡硬件（用户态门铃），不经过内核协议栈；
2. **零拷贝（Zero Copy）**：网卡 DMA 引擎直接读写应用注册的内存（显式 pin 住的 buffer），没有中间 buffer；
3. **CPU 卸载（Offload）**：传输可靠性（重传、排序、流控）由网卡硬件实现，CPU 只在建连时参与。

结果：**延迟降到 ~2 µs，CPU 占用≈0，带宽打满网卡线速**。RDMA 提供两类动词（verbs）语义：

- **Send/Recv**（双边）：收发双方都要提交请求，类似消息传递；
- **Read/Write**（单边）：本端直接读写**远端内存**，对端 CPU 完全无感——这是 NVSHMEM `put/get`（15.4）的底层原型。

> 类比：TCP 像"寄快递要经过邮局柜台逐件登记"；RDMA 是"你有一把我家仓库的钥匙，自己开门放东西，我甚至不用在家"。

## 3. InfiniBand：为 RDMA 而生的原生网络

**InfiniBand（IB）** 是从零为 RDMA 设计的专用网络（不是以太网），AI 训练集群的事实标准之一：

| 代 | 单端口速率 | 有效带宽 | 时代 |
|---|---|---|---|
| HDR | 200 Gb/s | 25 GB/s | A100 集群 |
| **NDR** | **400 Gb/s** | **50 GB/s** | **H100/B200 集群主流** |
| XDR | 800 Gb/s | 100 GB/s | Rubin 世代 |

IB 的三个核心特性：

1. **原生无损（Lossless）**：基于信用（credit-based）的链路级流控——接收方没 buffer 就不发，**网络不丢包**。RDMA 的可靠性设计假设底层无损，丢包重传对 RDMA 是灾难性的性能悬崖；
2. **SHARP 网内计算**：IB 交换机可以做归约（15.2 §5），AllReduce 的聚合在交换机里完成；
3. **专用生态**：需要 IB 网卡（ConnectX 系列）+ IB 交换机（Quantum 系列）+ 专用线缆——全套 NVIDIA（Mellanox）体系，性能好但封闭且贵。

## 4. RoCE v2：在以太网上跑 RDMA

**RoCE（RDMA over Converged Ethernet）v2** = 把 RDMA 协议封装进 UDP/IP，跑在标准以太网交换机上。

- **为什么存在**：以太网生态开放、便宜、运维人才多——云厂商（AWS、Meta、字节等）倾向用以太网建集群；
- **代价**：以太网默认**有损**，而 RDMA 要求无损，所以 RoCE 需要把以太网改造成无损：开启 **PFC**（Priority Flow Control，按优先级暂停）+ **ECN**（显式拥塞通知）+ 精心调参。这套配置的复杂度是 RoCE 的主要痛点（PFC 配置不当会引发队头阻塞甚至死锁）；
- **IB vs RoCE 选型**：IB 开箱即无损、延迟略低、SHARP 成熟；RoCE 开放便宜、规模上限大。头部 AI 集群两者都有大规模部署。

## 5. GPUDirect：最后一公里——网卡直接碰显存

RDMA 解决了"CPU 不碰数据"，但还差一步：默认情况下网卡 DMA 的对象是**主存**，GPU 数据仍要 `cudaMemcpy` 到主存再发。**GPUDirect 家族**补齐这最后一公里：

| 技术 | 解决什么 | 数据路径 |
|---|---|---|
| **GPUDirect P2P** | 节点内 GPU↔GPU | 经 NVLink/PCIe 直写对端显存（14.2 §6） |
| **GPUDirect RDMA** | 跨节点 网卡↔显存 | **网卡 DMA 直接读写 GPU 显存，不碰主存** |
| **GPUDirect Storage** | 存储↔显存 | NVMe 直接 DMA 进显存（训练数据加载） |
| **GPUDirect Async（IBGDA）** | 谁发起 | GPU kernel 直接给网卡下任务（15.4 §3） |

GPUDirect RDMA 前后的路径对比（发送方向）：

```
无 GPUDirect：GPU 显存 → cudaMemcpy → 主存 pinned buffer → 网卡 → 网络
有 GPUDirect：GPU 显存 ──────────（网卡 DMA 直读）──────────→ 网卡 → 网络
```

前提条件：网卡与 GPU 在 PCIe 拓扑上足够近（最好同 PCIe switch/同 NUMA，见 14.4），且驱动注册显存给网卡（`nvidia-peermem` 模块）。NCCL 的 NET transport（15.3 §5）自动完成这一切。

## 6. 一个数字感受层级落差

把一份 1 GB 的梯度从一台机器搬到另一台机器（400G NDR 网卡，50 GB/s 有效）：

```
跨节点：1 GB ÷ 50 GB/s ≈ 20 ms
节点内（NVLink 900 GB/s）：1 GB ÷ 900 GB/s ≈ 1.1 ms
```

**同一份数据，出不出机箱差 18 倍。** 这就是为什么 15.2 的分层算法要把跨节点流量压缩到 1/8，为什么 16.5 的映射规则是"通信密集的并行维度留在 NVLink 域内"。

## 7. 与全书主线的呼应

- **搬运主线**的集群段：`ld.global`（线程搬）→ TMA（SM 内 DMA）→ GPUDirect RDMA（网卡 DMA）→ IBGDA（kernel 直接驱动网卡）——"搬运职责不断从计算单元剥离"这条线在集群尺度的延续。
- **卸载思想**：RDMA 把协议栈从 CPU 卸载到网卡，与 TMA 把地址生成从 SM 卸载到专用引擎（Part 5）、SHARP 把归约卸载到交换机（15.2）是同一个哲学：**凡是模式固定的操作，都值得固化进离数据最近的硬件**。
- 下游：15.3（NCCL NET transport）、15.4（IBGDA）、16.5（拓扑映射）。

## 8. 参考资料

- [NVIDIA GPUDirect 官方文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [InfiniBand Trade Association 规范](https://www.infinibandta.org/)；NVIDIA Quantum-2（NDR 400G）平台资料
- RoCE v2：IETF RFC 4791（iWARP 对照）与 Mellanox RoCE 部署指南（PFC/ECN 调参）
- [NCCL 文档：跨节点传输与 GPUDirect RDMA](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
