# 15.1 集合通信算子：AllReduce / AllGather / ReduceScatter / AllToAll / Broadcast

> 核心问题：每个算子搬什么数据、谁是它的使用方？
>
> 阅读对象：通信零基础。读完这一篇，你应该能看着任何一份分布式训练代码里的 `all_reduce` / `all_gather` / `all_to_all`，立刻说出"数据从谁到谁、每份数据被搬了几次、为什么要搬"。

## 0. 为什么需要"通信算子"这个东西

卷一卷二（Part 1~13）讲的是**一颗 GPU 内部**的故事：SM 怎么调度、数据怎么在 HBM→L2→SMEM→寄存器之间流动。但大模型训练的基本事实是：**一张卡装不下模型，也算不动数据**——于是问题从"一颗 GPU 怎么跑得快"变成了"**很多颗 GPU 怎么一起算同一件事**"。

多 GPU 协同必然涉及数据交换。如果让每个工程师自己用 `send`/`recv` 手写这些交换，会得到一万种写法、一万种 bug，和一万种慢法。所以行业把多卡间最常见的几种数据交换模式**固化成标准算子**，就像 BLAS 把矩阵乘固化成 `gemm` 一样——这就是**集合通信算子（Collective Operations）**。

"集合（Collective）"这个词的意思是：**一组进程/GPU 共同参与、共同完成**的一次操作，区别于一对一的"点对点（Point-to-Point, P2P）"通信。

> 类比：一个班级（N 个同学）要做小组作业。
> - 老师把题目发给全班 = **Broadcast**
> - 每个人算出自己那部分结果，汇总给课代表求和 = **Reduce**
> - 全班每个人都要拿到"所有人结果的求和" = **AllReduce**
> 集合通信就是把这些"班级协作动作"标准化。

## 1. 基本术语：先把黑话说清楚

| 术语 | 含义 |
|---|---|
| **Rank** | 参与通信的每个进程（通常 = 每张 GPU）的编号，`0 ~ N-1` |
| **World Size（N）** | 参与通信的 GPU 总数 |
| **Communicator（通信子）** | 一组 Rank 的集合 + 它们之间的通信上下文。可以建子集（如"同一台机器的 8 张卡"是一个通信子） |
| **Root** | 某些算子里"特殊的那个 Rank"（如 Broadcast 的发送方） |
| **Buffer（缓冲区）** | 每个 Rank 上参与通信的显存区域，算子的输入输出都落在显存里 |
| **In-place / Out-of-place** | 输出是否覆盖输入缓冲区 |

一次集合通信的发起（以 NCCL 为例）长这样：

```cpp
// 每个 rank 都调用同一个函数，效果是整个通信子协同完成
ncclAllReduce(
    sendbuff, recvbuff,   // 输入/输出显存地址（本 rank 的）
    count,                // 元素个数
    ncclFloat32,          // 数据类型
    ncclSum,              // 归约操作：求和
    comm,                 // 通信子（哪些 rank 参与）
    stream                // 在哪个 CUDA Stream 上执行（异步！）
);
```

注意三点，这三点是理解后续所有内容的关键：

1. **每个 Rank 调用的是同一个函数**——"集合"语义由库内部协调，不是某个 Rank 当领导发号施令（虽然实现上常有逻辑上的 root）。
2. **通信发生在显存与显存之间**——send/recv buffer 都是 device pointer，数据不经过主机内存（这是 GPUDirect 的意义，见 Part 14.3）。
3. **通信是 Stream 上的异步操作**——它和 kernel 一样排在 CUDA Stream 里，这为 Part 15.5 的"通信-计算重叠"埋下伏笔。

## 2. 基本原语：四个"单人动作"

以下四个算子都有一个特殊的 **Root**。假设 World Size N = 4，每格 `[x]` 表示一个数据块：

### 2.1 Broadcast（广播）：一个人有，所有人都要有

```
Root(rank 0):  [A]  ──┬──> rank 0: [A]
                      ├──> rank 1: [A]
                      ├──> rank 2: [A]
                      └──> rank 3: [A]
```

- **数据量**：Root 发出 M，其余每个 Rank 收到 M。
- **使用方**：训练开始时同步初始权重、同步随机种子状态、推理时把输入 prompt 分发给所有 TP Rank。

### 2.2 Reduce（归约）：人人都有，汇总到一个人

```
rank 0: [a0] ──┐
rank 1: [a1] ──┼──> Root: [a0+a1+a2+a3]   （以 Sum 为例）
rank 2: [a2] ──┤
rank 3: [a3] ──┘
```

- **归约操作（Reduce Op）**：Sum / Max / Min / Prod / Avg（Avg 常由 Sum + 除以 N 实现）。要求是**满足结合律和交换律**的运算，否则无法并行归约。
- **使用方**：早期的参数服务器模式（梯度汇总到 server）。现代大规模训练里单点 Reduce 已少见（它是单点瓶颈，见 15.2），但它是理解 AllReduce 的必经一步。

### 2.3 Gather（收集）：每个人的东西都给 Root

```
rank 0: [a0] ──┐
rank 1: [a1] ──┼──> Root: [a0 | a1 | a2 | a3]   （拼接，不运算）
rank 2: [a2] ──┤
rank 3: [a3] ──┘
```

- 与 Reduce 的区别：**Gather 是拼接（concat），不做运算**，Root 收到的数据量是 N × M。

### 2.4 Scatter（分发）：Root 的东西切开分给大家

```
Root: [a0 | a1 | a2 | a3] ──┬──> rank 0: [a0]
                            ├──> rank 1: [a1]
                            ├──> rank 2: [a2]
                            └──> rank 3: [a3]
```

- Broadcast 是"完整复制"，Scatter 是"切开发放"，正好互为逆操作。

## 3. 复合算子：训练里真正干活的四个

单点原语的共同问题是 **Root 是瓶颈**（它一个人收/发 N 份）。复合算子去掉了 Root：**人人平等，人人出力**。

### 3.1 AllReduce（全归约）：集合通信的"第一主角"

> **每个人都贡献一份数据，归约后每个人都拿到完整结果。**

```
rank 0: [a0] ┐
rank 1: [a1] ├─ AllReduce(Sum) ─> 每个 rank 都得到 [a0+a1+a2+a3]
rank 2: [a2] ┤
rank 3: [a3] ┘
```

- **使用方：数据并行（DP）的梯度同步**。N 张卡各自用不同数据算出了 N 份梯度，要更新同一份模型，就必须保证大家用相同的梯度——AllReduce(Sum) 后再除以 N 即得平均梯度。这是深度学习里出现频率最高的集合通信，没有之一。
- **通信量**（每个 Rank 的发送+接收总量，Ring 算法，推导见 15.2）：

\[
V_{\text{AllReduce}} = \frac{2(N-1)}{N} \cdot M \;\approx\; 2M
\]

即：**每卡大约搬 2 倍于自己数据量的流量，且与卡数 N 几乎无关**——这是 AllReduce 能扩展到万卡的根本原因，15.2 会完整推导这个"魔术数字"。

### 3.2 ReduceScatter（归约并分散）：求和，但每人只留一片

> **先做 Reduce，再把结果切成 N 片，第 i 片给 Rank i。**

```
输入:  每 rank 有完整的 [x0|x1|x2|x3]（内容各不相同）
输出:  rank 0 拿到 [Σx0]，rank 1 拿到 [Σx1]，rank 2 拿到 [Σx2]，rank 3 拿到 [Σx3]
```

- **通信量**（Ring）：

\[
V_{\text{RS}} = \frac{N-1}{N} \cdot M \;\approx\; M
\]

- **使用方：FSDP / ZeRO 的梯度同步**（Part 16.1）。每张卡只负责更新模型的 1/N 参数，所以它只需要这 1/N 参数对应的梯度——用 ReduceScatter 而不是 AllReduce，省一半流量。

### 3.3 AllGather（全收集）：每人出一片，拼出完整版

> **每个 Rank 贡献自己的一片，最后人人拿到 N 片的完整拼接。**

```
输入:  rank i 有 [xi]（大小 M/N）
输出:  每 rank 都有 [x0|x1|x2|x3]（大小 M）
```

- **通信量**（Ring）：与 ReduceScatter 对称，≈ M。
- **使用方：FSDP / ZeRO 的前向与反向**——参数被切存在 N 张卡上，算到某一层时先 AllGather 把该层权重"临时拼完整"，用完即扔。
- **关键洞察**：`AllReduce ≈ ReduceScatter + AllGather`。先各自归约出 1/N，再拼接给所有人，两阶段各搬 ≈M，合计 ≈2M——这正是 3.1 那个公式的由来，也是 Ring AllReduce 的实现方式。

### 3.4 AllToAll（全互换）：人人给人人发快递

> **每个 Rank 把自己的数据切成 N 片，第 j 片发给 Rank j；同时收到来自所有 Rank 发给自己的那一片。**

```
rank 0: [a→0 | a→1 | a→2 | a→3]     （a→j 表示"发给 rank j 的那片"）
rank 1: [b→0 | b→1 | b→2 | b→3]
         ...  交换后 ...
rank 0 拿到: [a→0 | b→0 | c→0 | d→0]
```

- **通信量**：每卡发出 (N-1)/N · M，收到同样多。
- **使用方：MoE 的专家并行（EP）**（Part 16.4）。每张卡持有不同的"专家"，token 要按路由结果"寄"给对应专家所在的卡，算完再"寄"回来——一去一回两次 AllToAll。它是所有集合算子里**对网络最苛刻**的一个：N 个节点两两同时互发，任何一条链路慢都会拖住整体，这也是 MoE 训练对网络拓扑敏感的原因。

### 3.5 Barrier（屏障）

无人发数据，只是**所有 Rank 到齐后才放行**。GPU 世界里的角色类似 Part 13 的 Grid-wide sync，调试与性能测量时用来对齐时间线。

### 3.6 Send / Recv（点对点）

不属于"集合"，但集合算子的底层实现全是它。**Pipeline 并行**（Part 16.3）的层间激活传递用的就是 Send/Recv：第 i 段的输出 Send 给第 i+1 段。

## 4. 汇总：一张表记住所有算子

| 算子 | 一句话语义 | 每卡通信量（Ring，M=单卡数据大小） | 主要使用方 |
|---|---|---|---|
| Broadcast | 1 → 所有人 | Root 发 M，其余收 M | 参数/输入同步 |
| Reduce | 所有人 → 1（带运算） | Root 收 N·M | 参数服务器（已少用） |
| **AllReduce** | 所有人 → 所有人（带运算） | **≈ 2M** | **DP 梯度同步** |
| **ReduceScatter** | 归约 + 每人留 1/N | **≈ M** | **FSDP/ZeRO 反向** |
| **AllGather** | 每人出 1/N 拼完整 | **≈ M** | **FSDP/ZeRO 前向**、TP |
| **AllToAll** | 人人互换切片 | ≈ M（两两互发） | **MoE 专家并行** |
| Barrier | 对齐，不搬数据 | 0 | 调试/计时 |
| Send/Recv | 一对一 | 双方各 M | **Pipeline 并行** |

> 直觉记忆法：**ReduceScatter 和 AllGather 互为逆操作，各搬一份 M；AllReduce 是它俩的组合，搬两份 M；AllToAll 也是一份 M，但链路模式是 N×N 全互联，对网络公平性要求最高。**

## 5. 进阶问题（预告后续章节）

学完这一篇，自然的疑问是：

1. 这些"≈ M / ≈ 2M"是怎么算出来的？为什么 Ring 能做到与 N 无关？→ **15.2 通信算法**
2. 这些算子在 GPU 上是谁执行的？CUDA kernel 吗？SM 吗？→ **15.3 NCCL 内部**
3. 能不能让 kernel 在计算途中直接读写别的卡的显存，而不走"调用集合算子"这个流程？→ **15.4 NVSHMEM**
4. 通信时 SM 在干嘛？能不能一边算一边通信？→ **15.5 通信-计算重叠**
5. 这些算子在不同并行策略里到底怎么组合使用？→ **Part 16 并行策略**

## 6. 参考资料

- [NCCL 官方文档：Collective Operations 语义定义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [MPI 标准](https://www.mpi-forum.org/)：集合通信算子的术语体系源自 MPI，NCCL 语义与 MPI 基本对齐
- Part 16.1（DP/FSDP/ZeRO）、16.4（EP/MoE）：各算子的"使用现场"
