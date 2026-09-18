# 14.4 拓扑与亲和：rail-optimized、NUMA 与拓扑探测

> 核心问题：通信任务应该放到哪些 GPU、哪张网卡上才最快？
>
> 前置：14.1~14.3（三级互联）、15.2（通信算法）。本篇是全书的"地图课"——硬件给了你不均匀的带宽，软件（NCCL、训练框架、调度器）必须**看懂地图**才能把通信放到快的路上。

## 0. Why：带宽是不均匀的，"放哪"就是性能

前三篇反复出现一个事实：GPU 间带宽取决于**走哪条路**——NV-HBI（10 TB/s）> NVLink（1.8 TB/s）> PCIe（128 GB/s）> IB（50 GB/s）。于是集群性能问题的一半是：

> **把通信量大的任务对，放到带宽大的链路上；把每张 GPU 的流量，引到离它最近的网卡上。**

这件事叫**拓扑感知（Topology Awareness）/ 亲和（Affinity）**。本篇分三层讲：怎么看懂一台机器的拓扑（§1~2）、集群网络怎么设计（§3~4）、软件如何利用拓扑（§5）。

## 1. 看懂单机拓扑：`nvidia-smi topo -m`

第一台要上手的工具是 `nvidia-smi topo -m`，它打印 GPU 两两之间、以及 GPU 与网卡/CPU 之间的"距离"：

```
        GPU0  GPU1  GPU2  GPU3  NIC0  NIC1  CPU Affinity
GPU0     X    NV8   NV8   NV8   PIX   NODE  0-31
GPU1    NV8    X    NV8   NV8   NODE  PIX   0-31
GPU2    NV8   NV8    X    NV8   NODE  NODE  32-63
GPU3    NV8   NV8   NV8    X    NODE  NODE  32-63
NIC0    PIX   NODE  NODE  NODE   X    NODE
NIC1    NODE  PIX   NODE  NODE  NODE   X
```

图例（从近到远，带宽递减）：

| 标记 | 含义 | 带宽直觉 |
|---|---|---|
| `NV#` | 经 # 条 NVLink 直连（`NV8` = 8 条 bond） | 最肥，900 GB/s 级 |
| `PIX` | 经同一个 PCIe switch | 较好 |
| `PXB` | 经多个 PCIe switch（同主机） | 中等 |
| `PHB` | 经 CPU 根复合体（同 NUMA 节点） | 偏差 |
| `NODE` | 同 NUMA 节点内（经主存一致性域） | 差 |
| `SYS` | **跨 NUMA 节点**（要过 CPU 间互联 UPI） | 最差，务必避免 |

**读表要点**：GPU0 与 NIC0 是 `PIX`（同 PCIe switch）→ GPU0 的跨节点流量应该走 NIC0；若错走 NIC1（`NODE`），GPUDirect RDMA 可能退化或多绕一跳，实测带宽掉 20~50%。

配套命令：`nvidia-smi nvlink -s`（看每条 NVLink 的状态与速率）、`ibstat` / `ibv_devinfo`（看 IB 网卡端口状态与速率）。

## 2. NUMA 亲和：CPU 侧的位置同样重要

GPU 服务器的 CPU 通常是**双路（两个 socket）**，每个 socket 是一个 **NUMA 节点**：各自挂本地内存、本地 PCIe 插槽（一部分 GPU 和网卡）。跨 socket 访问要走 CPU 间互联（UPI/Infinity Fabric），延迟翻倍、带宽砍半。

实践规则：

1. **进程绑核**：训练进程绑定到"它的 GPU 所在 NUMA 节点的 CPU 核"（`numactl --cpunodebind` / `--membind`），避免控制面跨 socket；
2. **网卡归属**：每张 GPU 有"本命网卡"（同 NUMA/同 PCIe switch 的那张），NCCL 会自动选（`NCCL_IB_HCA` 可人工指定）；
3. **内存分配**：Host 侧 buffer（如 pinned memory）分配在本 NUMA 节点。

> 一个典型的性能事故：`nvidia-smi topo -m` 里 GPU3 与 NIC0 是 `SYS`，而启动脚本把 GPU3 的流量绑到了 NIC0——跨 NUMA 的 RDMA 让 AllReduce 带宽腰斩。排障第一步永远是先看这张表。

## 3. 集群网络拓扑：Fat-Tree 与 Rail-Optimized

### 3.1 Fat-Tree（胖树）：通用无阻塞结构

数据中心经典拓扑：接入层（leaf）→ 汇聚/核心层（spine），上行带宽逐层加粗，理想情况下**任意两台服务器间等带宽、无阻塞**（1:1 收敛比）。

- 实际集群常用 **收敛比** 描述：leaf 下行 32×400G、上行 16×400G = 2:1 超订（oversubscribed）——省钱，但跨 leaf 大流量时会抢带宽；
- AI 训练对"全员同时通信"（AllReduce）极其敏感，超订比是训练集群网络的第一指标。

### 3.2 Rail-Optimized（轨道优化）：AI 集群的主流答案

观察：8 卡机的 AllReduce 环里，**每张卡只需要和"其他机器的同一个位置"通信**（GPU i ↔ 别机的 GPU i）。于是把网络组织成 8 条独立"轨道（rail）"：

```
轨0: GPU0(机1) ─ NIC0 ─ Leaf0 ─┬─ Spine ─┬─ Leaf0 ─ NIC0 ─ GPU0(机N)
轨1: GPU1(机1) ─ NIC1 ─ Leaf1 ─┤         ├─ Leaf1 ─ NIC1 ─ GPU1(机N)
...                             └─────────┘
轨7: GPU7(机1) ─ NIC7 ─ Leaf7 ─┴─────────┴─ Leaf7 ─ NIC7 ─ GPU7(机N)
```

- 每轨是一个独立的小 fat-tree，**轨间不互通**（或弱互通）；
- 好处：AllReduce 的 Ring 可以完美落在轨内（每卡只用自己的 NIC），流量模型可预测、无跨轨干扰；扩集群就是加机器进每一轨；
- 代价：AllToAll（MoE/EP，15.1 §3.4）的 N×N 流量不完全贴合轨道——这是 EP 大组需要 NVL72 大 NVLink 域或专门网络设计的原因（16.4/16.5 展开）。

## 4. 拓扑探测：软件如何"看见"地图

硬件拓扑不会自动变成软件决策，中间靠探测：

1. **单机内**：NCCL 启动时解析 PCIe/NVLink 拓扑（`nvidia-smi topo -m` 同款信息 + NVML API），建图、搜环（15.3 §2）；可用 `NCCL_TOPO_DUMP_FILE=topo.xml` 导出检查；
2. **跨节点**：各 Rank 交换本机拓扑，合并成全局图，再决定 Ring/Tree 的边落在哪些链路上；
3. **调度层**：集群调度器（Kubernetes device plugin、Slurm 的 topology plugin）按拓扑分配整机/整轨资源，避免"一个 8 卡任务被切到两台 4 卡的机器上"这类碎片化。

**验证工具链**：

```bash
nvidia-smi topo -m          # 单机拓扑矩阵
nvidia-smi nvlink -s        # NVLink 链路状态
ibstat                      # IB 网卡速率/状态
nccl-tests (all_reduce_perf) # 端到端带宽实测：busbw 应接近理论值
```

`nccl-tests` 的 **busbw（算法带宽）** 是黄金指标：它把测得的 AllReduce 时间按 15.2 的公式折算回"每卡有效带宽"，直接反映拓扑利用是否健康。

## 5. 映射规则：并行策略 → 拓扑（Part 16.5 预告）

把本篇和 Part 15 合起来，得到那张经验法则表：

| 通信模式 | 流量特征 | 应放置的位置 |
|---|---|---|
| TP 的 AllReduce（每层 2 次，关键路径） | 高频、延迟敏感 | **NVLink 域内**（节点内或 NVL72 机柜内） |
| EP 的 AllToAll（MoE dispatch/combine） | N×N 全互联 | 尽量 NVLink 域内；否则 rail 内 |
| PP 的 Send/Recv（层间激活） | 点对点、量小 | 跨节点可接受 |
| DP/FSDP 的梯度同步（可重叠，15.5） | 量大但可藏 | **跨节点 IB/RoCE** |

一句话：**延迟敏感的留在近处，带宽敏感但可重叠的放远处**——和 Part 1 存储层次的思想（快的近、慢的远，用异步掩盖慢）完全同构，只是尺度从"一颗芯片"放大到了"一个集群"。

## 6. 与全书主线的呼应

- **存储层次的集群版**：NV-HBI / NVLink / IB 就是集群尺度的 寄存器 / SMEM / HBM——"层次化 + 用局部性换性能"是同一个第一性原理；
- **Roofline 的集群版**：单机 Roofline 的横轴是算术强度（FLOP/Byte），集群版的"带宽屋顶"取决于通信落在哪一层拓扑上；
- 下游：15.2 §4（分层算法）、15.3（NCCL 拓扑探测实现）、16.5（混合并行映射的完整推导）。

## 7. 参考资料

- `nvidia-smi` 手册（topo / nvlink 子命令）；[NCCL 拓扑文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA Rail-Optimized 网络设计指南（DGX SuperPOD 参考架构）](https://www.nvidia.com/en-us/data-center/dgx-superpod/)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests)：busbw 指标的定义与实测方法
- Meta *RoCE Networks for AI Training at Scale*（RoCE 大规模部署实践）
