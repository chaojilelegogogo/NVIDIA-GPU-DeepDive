# 15.2 通信算法：Ring / Tree / Recursive Halving-Doubling 与带宽-延迟模型

> 核心问题：同一个 AllReduce，为什么 Ring 和 Tree 在不同规模下各占优？
>
> 前置：15.1（算子语义）。读完这一篇，你应该能自己推导出 Ring AllReduce 的通信量公式，并解释"为什么万卡集群上 AllReduce 的时间几乎不随卡数增长"。

## 0. 先建立度量工具：α-β 模型

评价一个通信算法，不能像卷二那样只看"搬了多少字节"，因为网络通信有两笔开销：

\[
T(\text{消息}) = \alpha + \beta \cdot n
\]

- **α（latency，延迟项）**：发一条消息本身要付的固定代价——协议栈处理、链路传播、内核/硬件门铃等。与消息大小无关，一次握手就是一份 α。GPU 间典型值：NVLink 域内约 1~2 µs，跨节点 RDMA 约 2~5 µs。
- **β（bandwidth 项）**：每字节传输时间 = 1/带宽。如 400 Gb/s IB 网卡，β = 1/(50 GB/s)。
- **n**：消息字节数。

于是一个通信算法的好坏由两个量决定：

1. **走了几步（step 数）** → 决定 α 的总开销（每步至少一个 α）；
2. **每卡总共搬了多少字节** → 决定 β 的总开销。

**小消息被 α 主导**（步数少的算法赢），**大消息被 β 主导**（每卡总流量小的算法赢）。这个"两分钱"模型是理解 Ring vs Tree 之争的钥匙。

## 1. 朴素方案：为什么"都发给一个人"不行

先看最直觉的 AllReduce 实现——**参数服务器式**：所有人把梯度发给 Rank 0，Rank 0 求和后再 Broadcast 回来。

- **步数**：看似只有 2 大步（收 + 发）；
- **流量**：Rank 0 要收 N·M 字节、发 N·M 字节。**它的网卡成了单点瓶颈**：N = 1024 时，其余卡每人只搬 2M，Rank 0 却要搬 2048M——木桶效应，整体时间由最慢的那个决定。
- **结论**：集合通信的第一设计原则是**负载均衡——每个 Rank 的收/发流量必须相等**。所有正经算法（Ring/Tree/RHD）都是为了满足这一点。

## 2. Ring：大消息场景的王者

### 2.1 环怎么排

把 N 个 Rank 排成一个逻辑环（0→1→2→…→N-1→0），每个 Rank 只和**左右两个邻居**通信。注意：环是**逻辑**的，物理上 NCCL 会尽量让相邻 Rank 对应物理上相邻的链路（NVLink 环、 rail 拓扑，见 Part 14.4）。

### 2.2 Ring AllReduce = Ring ReduceScatter + Ring AllGather

把每卡的数据切成 N 片。以 N = 4、每卡数据 [x0|x1|x2|x3] 为例。

**阶段一：ReduceScatter（N-1 步）**——目标：第 i 片的全量求和最终落在 Rank i 手里。

- 第 1 步：Rank i 把"自己手上的第 (i−1) 片"发给右邻居，同时收到左邻居发来的一片，**与本地对应片相加**。
- 第 2 步：把刚加完的那一片继续往右传，再与下一个 Rank 的对应片相加……
- 走 N−1 步后，每一片都恰好绕过了除"持有者"外的所有 Rank，累加了所有人的贡献。

**阶段二：AllGather（N-1 步）**——目标：把各自手里已完成的 1/N 结果分发给所有人。

- 同样的环，只是把"相加"换成"转发"：每步把自己已有的完整片发给右邻居，同时收下一片。
- N−1 步后，每卡都有全部 N 片 = 完整结果。

### 2.3 通信量推导（重要，请自己推一遍）

每个阶段：N−1 步，每步每卡发送 1 片 = M/N 字节。

\[
V_{\text{每阶段}} = (N-1)\cdot\frac{M}{N}
\qquad\Longrightarrow\qquad
V_{\text{AllReduce}} = \underbrace{\frac{(N-1)M}{N}}_{\text{ReduceScatter}} + \underbrace{\frac{(N-1)M}{N}}_{\text{AllGather}} = \frac{2(N-1)}{N}M \approx 2M
\]

**两个惊人的性质**：

1. **与 N 无关**：N 从 8 涨到 8192，每卡流量从 1.75M 涨到 ≈2M——几乎不变。这就是"万卡 AllReduce 可行"的数学基础。
2. **带宽最优**：可以证明，AllReduce 每卡流量下界就是 2(N−1)M/N（每片数据至少要离开来源卡一次、到达每卡一次），Ring 打到了理论下界。

### 2.4 代价：步数

Ring 的软肋在 α 项：**总步数 2(N−1)，随 N 线性增长**。N = 8192 时约 1.6 万步，若每步 α = 2 µs，仅延迟项就 ≈ 33 ms——小消息下完全不可接受。

> **Ring 的画像：β 最优、α 随 N 线性恶化 → 适合大消息，不适合小消息。**

## 3. Tree / Recursive Halving-Doubling：小消息的救星

### 3.1 Recursive Halving-Doubling（递归减半-倍增，RHD）

思想来自二分归约：每步 Rank 两两配对、各发一半数据。

- **ReduceScatter 阶段**（递归减半）：第 k 步，配对的两个 Rank 交换"对方负责的那一半数据"并归约。log₂N 步后，每卡持有 1/N 的结果。
- **AllGather 阶段**（递归倍增）：逆过程，每步把已有数据翻倍，log₂N 步后人人有全量。

\[
\text{步数} = 2\log_2 N, \qquad V \approx 2M \ \text{（与 Ring 同阶的带宽最优）}
\]

N = 8192 时步数只有 26 步（Ring 是 1.6 万步）。**α 从 O(N) 降到 O(log N)**。代价是实现复杂（要求 N 为 2 的幂，否则要退化处理），且每步的配对关系对物理拓扑敏感。

### 3.2 Tree（树形）

- 用一棵（或两棵）树做 Reduce 到根、再 Broadcast 下去，步数 2·log₂N。
- 经典问题：树的**非叶子节点流量不均衡**（越靠根越忙）。解决办法是 **Double Binary Tree**（双二叉树，NCCL 默认之一）：构建两棵互补的树，每个节点在其中一棵树是叶子、另一棵是中间节点，两棵树各传一半数据——流量重新均衡，且保留 O(log N) 步数。

> **Tree/RHD 的画像：α 为 O(log N)、β 同阶最优 → 适合小-中消息。**

## 4. 分层（两阶段）算法：万卡集群的真实形态

真实集群的带宽是**分层**的（Part 14）：节点内 NVLink ≈ 900 GB/s（B200 1800 GB/s），节点间 IB 每卡只有 ≈ 50 GB/s——差一个数量级以上。直接把 8192 张卡排成一个环，环上就会出现"跨节点慢边"卡死整体。

**两阶段 AllReduce**（hierarchical / multi-rail）的思路：快慢分开。

```
阶段1 节点内 ReduceScatter：8 卡 NVLink 域内归约，每卡留 1/8       （走快车道）
阶段2 跨节点 AllReduce：   每台机器的"代表卡"们对 1/8 数据做 AllReduce（走慢车道，但数据量只剩 1/8）
阶段3 节点内 AllGather：   把结果在 NVLink 域内拼回 8 卡              （走快车道）
```

- 跨节点流量被压缩到 **M/8**（每卡只在阶段 2 参与），慢链路的负担减少 8 倍；
- 这就是 NCCL 自动做的事（`NCCL_ALGO` 内部会结合拓扑选 Ring/Tree/CollNet），也是"为什么 TP 放节点内、DP 放节点间"这条经验法则的物理基础（Part 16.5 展开）。

## 5. In-Network Reduction：把归约搬进交换机

上面所有算法都假设"交换机只转发，不计算"。NVIDIA 的 **SHARP**（Scalable Hierarchical Aggregation and Reduction Protocol）打破了这个假设：

- **NVSwitch SHARP / NVLS**（NVLink SHARP，Hopper 起支持）：节点内 8 卡 AllReduce 直接在 NVSwitch 芯片里完成归约。每卡只需把数据发进交换机、再从交换机收回结果——**节点内 AllReduce 每卡流量从 2M 降到 M×2 次单向（发 M + 收 M），等效带宽翻倍**，且 SM 参与更少。
- **IB SHARP**：InfiniBand 交换机在树形聚合点做归约，跨节点 AllReduce 的收敛点从"某张卡"挪到"交换机"，消除单点。

这是"算力主线"在网络侧的翻版：**和 Tensor Core 把 MMA 固化进芯片一样，SHARP 把 Reduce 固化进了交换机**。

## 6. 选型决策表（实战结论）

| 场景 | 赢家 | 原因 |
|---|---|---|
| 大消息（≥ MB 级）、卡数多 | **Ring** | β 主导，Ring 带宽最优且对 N 免疫 |
| 小消息（KB 级）、卡数多 | **Tree / Double Binary** | α 主导，O(log N) 步数 |
| 小消息、卡数少（单机 8 卡） | Ring 或 Tree 均可，差异小 | α、β 都小 |
| 节点内 + NVSwitch（Hopper+） | **NVLS（SHARP）** | 交换芯片代劳，SM 几乎不参与 |
| 跨节点万卡 | **分层两阶段** | 把慢链路流量压缩 8× |
| AllToAll（MoE） | 无通用最优，依赖拓扑 | N×N 全互联，看 Part 16.4/16.5 的拓扑映射 |

NCCL 的运行时选择逻辑（15.3 会细讲）本质上就是内置了这张表：根据 `消息大小 × 拓扑` 在 LL/LL128/Simple 协议 × Ring/Tree/NVLS 算法间切换。

## 7. 与全书主线的呼应

- **α-β 模型 ↔ Roofline**（Part 1 §1.6）：Roofline 用"算力 vs 带宽"两个屋顶判断 kernel，α-β 用"延迟 vs 带宽"两笔账判断通信算法——同一个"分清两笔开销"的思想。
- **流水线气泡**（Part 16.3）：PP 的气泡问题与 α 项问题同构——都是"固定开销在规模面前的摊薄问题"。
- **延迟隐藏**（Part 1 §1.1.2）：Ring 把大数据切成小块流水传输，本质就是通信侧的"用并行度换延迟暴露"。

## 8. 参考资料

- Thakur et al., *Optimization of Collective Communication Operations in MPICH*（RHD/递归倍增的经典论文）
- [NCCL 文档：Algorithms（Ring / Tree / CollNet / NVLS）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NVIDIA SHARP 官方介绍](https://docs.nvidia.com/networking/display/sharpv300)
- Baidu, *Deep Learning Training with Ring-Allreduce*（2017，把 Ring AllReduce 引入 DL 训练的标志性文章）
