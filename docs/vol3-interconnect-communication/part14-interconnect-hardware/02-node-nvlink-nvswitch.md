# 14.2 节点内互联：NVLink 1~5 代、NVSwitch 与 DGX/HGX 拓扑

> 核心问题：一台机器里的 8 张 GPU 如何全带宽互联？
>
> 前置：14.1（封装内）。本篇讲**一台服务器内部**的 GPU 互联——这是通信带宽最肥的一层，也是 TP（张量并行）为什么必须放在节点内的原因（Part 16.5）。

## 0. Why：PCIe 为什么不够用

2016 年之前，多 GPU 之间唯一的通路是 **PCIe**：GPU A → PCIe switch → GPU B。两个致命问题：

1. **带宽低**：PCIe Gen3 x16 双向合计约 32 GB/s——同期 V100 的显存带宽是 900 GB/s，差近 30 倍。GPU 间传一次梯度的时间够本地显存搬 30 遍。
2. **路径绕**：P2P 流量要过 PCIe switch 甚至 CPU 根复合体，延迟高、还挤占 CPU 的 PCIe 资源。

数据并行训练兴起后，"每步都要 AllReduce 梯度"让 GPU 间通信从偶发变成主负载——**PCIe 这条路必须被专用互联取代**。这就是 NVLink 的诞生背景（2016，Pascal P100）。

## 1. NVLink 五代演进：一张表

NVLink 是**点对点串行链路**，每张 GPU 集成若干条 link，每条 link 双向工作。整卡带宽 = link 数 × 单 link 速率 × 2（双向）：

| 代 | 搭载 GPU | 单 link 速率（双向） | link 数/卡 | **整卡 NVLink 带宽** | 同期 PCIe | 倍数 |
|---|---|---|---|---|---|---|
| NVLink 1（2016） | P100 | 40 GB/s | 4 | **160 GB/s** | Gen3 x16 ≈ 32 GB/s | 5× |
| NVLink 2（2017） | V100 | 50 GB/s | 6 | **300 GB/s** | Gen3 x16 ≈ 32 GB/s | 9× |
| NVLink 3（2020） | A100 | 50 GB/s | 12 | **600 GB/s** | Gen4 x16 ≈ 64 GB/s | 9× |
| NVLink 4（2022） | H100 | 50 GB/s | 18 | **900 GB/s** | Gen5 x16 ≈ 128 GB/s | 7× |
| NVLink 5（2024） | B200 | 100 GB/s | 18 | **1800 GB/s** | Gen5/6 x16 | 14× |

> 推导示例（A100）：12 link × 50 GB/s = 600 GB/s ✅ 与官方一致。B200：18 × 100 = 1800 GB/s ✅。**记公式：整卡带宽 = link 数 × 单 link 双向速率**——和 Part 1x 里显存带宽 = 位宽 × 引脚速率 是同一种乘法。

两个值得注意的细节：

- **NVLink 4→5 的翻倍靠的是单 link 速率翻倍**（50→100 GB/s，信号调制升级），link 数没变；
- **NVLink 对 PCIe 的代际压制稳定在 5~14 倍**——这就是"节点内通信尽量不出 NVLink 域"这条铁律的定量来源。

## 2. 没有 NVSwitch 的时代：直连拓扑的局限

NVLink 1/2 时代（P100/V100），GPU 之间是**直连**：link 直接焊在两张卡之间。问题随之而来——

以 V100 为例，6 条 link 要连接 8 张卡，**不可能全互联**（全互联需要每卡 7 条 link 且拓扑是 8 节点完全图）。实际产品（DGX-1V）用的是 **Hybrid Cube-Mesh**：

```
        GPU0 ──── GPU1
        ╱ │ ╲      ╱ │ ╲
     GPU7   GPU2  ...      （立方体的 8 个顶点 + 部分面对角线）
        ╲ │ ╱      ╲ │ ╱
        GPU6 ──── GPU5
```

- 任意两卡之间**要么直连（1 hop）、要么要经过中间卡转发（2 hop）**——带宽和延迟不均匀；
- 做 AllReduce 时，Ring 只能沿实际存在的边走，可用带宽取决于最弱的那条边。

**结论：直连拓扑的 link 数决定了互联度，而 link 数受封装引脚限制——需要一个"交换"角色来解耦。**

## 3. NVSwitch：把"直连"变成"全互联"

**NVSwitch 是一颗专用的 NVLink 交换芯片**：所有 GPU 的 NVLink 全部接到 NVSwitch 上，由它做任意 GPU 对之间的转发。效果：

- **任意两卡等带宽、等延迟**（逻辑全互联），不再有"几 hop"的概念；
- 8 卡同时两两通信也能各自跑满——NVSwitch 内部是无阻塞交换结构；
- 从 Hopper 起，NVSwitch 还能**计算**：NVLS（NVLink SHARP）在交换芯片内完成 AllReduce 归约（见 15.2 §5、15.3 §6）。

NVSwitch 自身的演进：

| 代 | 搭载 | 单芯片能力 | 系统形态 |
|---|---|---|---|
| NVSwitch 1（2018） | V100（DGX-2） | 16 GPU 全互联 | 首次出现，DGX-2 用 12 颗连 16 卡 |
| NVSwitch 2（2020） | A100（DGX A100） | 8 GPU 全互联 | 6 颗连 8 卡 |
| NVSwitch 3（2022） | H100（DGX H100） | + **SHARP 网内归约** | 4 颗连 8 卡 |
| NVSwitch 4（2024） | B200（GB200 NVL72） | + 更大基数 | **机柜级：72 张 GPU 一个 NVLink 域** |

## 4. DGX/HGX 8 卡节点：标准形态

以 DGX H100 为例，一台 8 卡节点的互联结构：

```
GPU0 ─┐
GPU1 ─┤
...   ├── 4 × NVSwitch 3 ──（任意 GPU 对之间 900 GB/s 全互联）
GPU7 ─┘
  │
  └─ 每卡另出：PCIe Gen5 → CPU（控制面）+ 400G IB 网卡 × 8（出节点，见 14.3）
```

- **HGX** 是 NVIDIA 卖给 OEM（戴尔、超微等）的 8 卡基板标准，DGX 是 NVIDIA 自有品牌整机——互联拓扑相同；
- 8 卡 + NVSwitch 构成的这个"全互联域"叫 **NVLink 域（NVLink Domain）**，是集群里最重要的边界：**域内 900 GB/s，域外 50 GB/s/卡（400G 网卡）——18 倍的落差**。

## 5. NVL72：NVLink 域扩展到整个机柜

Blackwell 世代最重要的系统创新：**GB200 NVL72 把 NVLink 域从 8 卡扩大到 72 卡**——用 9 个 NVSwitch 托盘（tray）通过铜缆背板连接 36 个 Grace-Blackwell 超级芯片（72 GPU），整个机柜内任意两 GPU 间 900+ GB/s（NVLink 5 时代为每 GPU 1.8 TB/s 聚合）。

- 意义：**TP=72、EP=72 的大并行组可以全部留在 NVLink 域内**，不必碰慢速的 IB 网络——直接改变了 Part 16 并行策略的映射规则（16.5 会展开）；
- NVLink 从此有了"出机箱"的形态（NVLink Switch System），但注意它仍然是**专用封闭网络**，与 14.3 的 IB/以太网是两套体系。

## 6. 软件视角：P2P 访问与拓扑感知

- 节点内 GPU 互访在 CUDA 里叫 **P2P（Peer-to-Peer）**：`cudaDeviceEnablePeerAccess` 后，kernel 可以直接 `ld`/`st` 另一张卡的显存（走 NVLink），NCCL 的 P2P transport（15.3 §5）就建立在这之上；
- 但"能访问"不等于"该访问"——是否经过 NVSwitch、与网卡的相对位置，决定了实际带宽。这正是 14.4（拓扑与亲和）的内容。

## 7. 与全书主线的呼应

- **协作范围主线**的硬件载体：Thread → Warp → Block → Cluster（节点内 SM 间）→ **NVLink 域（节点内 GPU 间）**→ 集群。NVLink/NVSwitch 是"协作范围"突破单卡的物理基础。
- **NVSwitch 做归约（SHARP）** = 算力主线在网络侧的翻版（计算下沉到数据经过的地方），与 TMA（搬运专用化）思想同源。
- 下游：15.2（Ring 沿 NVLink 边走）、15.3（NCCL 拓扑探测）、16.5（TP 放 NVLink 域内）。

## 8. 参考资料

- [NVIDIA NVLink 与 NVSwitch 官方页](https://www.nvidia.com/en-us/data-center/nvlink/)
- 各代 DGX 白皮书：DGX-1V（Hybrid Cube-Mesh）、DGX-2（NVSwitch 首秀）、DGX A100/H100、GB200 NVL72
- A100/H100/B200 手册（`vol1-gpu-hardware/gpgpu-models/`）中的 NVLink 规格表
