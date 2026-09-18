# 15.5 通信-计算重叠与通信 Kernel 化

> 核心问题：如何把通信藏进计算的缝隙里？
>
> 前置：15.1~15.4，Part 11（性能优化方法论）。读完这一篇，你应该能判断一个训练 step 里"哪些通信可以藏、哪些藏不住"，并理解"通信 kernel 化"这条进阶路线的动机。

## 0. 为什么重叠是可能的

回顾卷一的两个事实：

1. **通信的搬运主力不是计算单元**：跨节点走网卡 DMA（RDMA），节点内走 NVLink 的 copy engine 或 NCCL kernel 的 LD/ST——它们都不需要 Tensor Core 参与；
2. **GPU 是多队列设备**：不同 CUDA Stream 上的 kernel 可以并发执行（Part 8）。

所以理论上：把通信排到 `stream_comm`、计算排到 `stream_comp`，两者就能**时间重叠**——通信的这段时间不再是纯浪费。这和 Part 5/Part 2 的"访存与计算重叠"（`cp.async`、双缓冲）是同一个思想，只是对象从"SM 内的访存"换成了"GPU 间的通信"。

\[
T_{\text{step}} \approx \max(T_{\text{comp}},\ T_{\text{comm}}) \quad\text{（理想重叠）} \qquad vs \qquad T_{\text{comp}} + T_{\text{comm}} \quad\text{（串行）}
\]

**但有两个"税"让现实达不到理想式：**

- **SM 税**：节点内 NCCL kernel 要占 SM（15.3 §1），与计算 kernel 抢资源；
- **依赖税**：不是所有通信都"独立"——等它结果的计算只能干等。

这一篇的全部内容，就是围绕"如何少交这两种税"。

## 1. 最容易藏的通信：DP 梯度 AllReduce

数据并行的反向传播有一个天然的好性质：**第 L 层的梯度算完后，它的 AllReduce 与第 L−1 层的梯度计算无关**。于是：

```
反向计算 layer L   →  立刻发起 layer L 的 AllReduce（comm stream）
反向计算 layer L-1 →  同时 AllReduce(L) 在后台搬数
反向计算 layer L-2 →  同时 AllReduce(L-1) 在后台搬数
...
```

**梯度分桶（Gradient Bucketing）**：不为每层单独发 AllReduce（小消息会被 α 吃掉，15.2 §0），而是把若干层的梯度攒成一个 ~25 MB 的"桶"，桶满即发一次大 AllReduce——兼顾重叠与大消息带宽。PyTorch DDP 的 `bucket_cap_mb` 就是这个旋钮。

> 这是"通信重叠"性价比最高的一处：**纯软件技巧，零硬件要求，DP 场景几乎免费拿到**。

## 2. FSDP 的 prefetch：把 AllGather 提到计算前面

FSDP/ZeRO（Part 16.1）里，参数是分片存的，算第 L 层前要先 AllGather 出完整权重。藏法与 DP 对称：

```
计算 layer L   的同时   prefetch AllGather(layer L+1 的权重)
```

- 反向时同理，梯度用 ReduceScatter 在层间流水；
- 实现手段：`forward_prefetch` / `backward_prefetch` 选项，本质是提前把通信 enqueue 到 comm stream。

## 3. 最难藏的通信：TP 与 MoE

### 3.1 TP（张量并行）：通信在关键路径上

TP 的 AllReduce/AllGather 发生在**每一层的中间**（Part 16.2）：GEMM 的前半段结果不归约完，后半段没法开始——**通信在依赖链的关键路径上**，没有无关计算可用来填缝隙。

可行手段只剩两条：

1. **缩小单次通信量/延迟**：TP 放节点内 NVLink 域（Part 16.5 的映射规则即为此），用 NVLS 让交换芯片代劳（15.3 §6）；
2. **切得更碎做流水**：把 GEMM 按列切成多段，段间通信与前一段计算流水（DeepSpeed-Ulysses、Megatron 的 sequence-parallel overlap 属于这类思想）。

### 3.2 MoE 的 AllToAll：用"另一个 micro-batch"来填

MoE 的 dispatch/combine AllToAll 同样在关键路径上，工业界的标准解法是 **DualPipe / 交错调度**：

```
micro-batch A:  [dispatch A] [experts A] [combine A]
micro-batch B:        [experts B]     ← B 的计算故意排在 A 的通信窗口里
```

用另一个 micro-batch 的 expert 计算，盖住本 micro-batch 的 AllToAll——**通信没变快，但 GPU 没在等**。代价是实现复杂度（两份激活驻留显存）与调度器的精细控制。

## 4. 重叠的度量：通信暴露时间

评价重叠做得好不好的指标（Part 11 方法论在分布式场景的延伸）：

\[
\text{暴露通信时间} = T_{\text{step}} - T_{\text{纯计算}}
\]

- 用 Nsight Systems 看时间线：comm stream 上的 NCCL kernel 与 comp stream 的计算 kernel 是否真的在时间轴上重叠；
- 目标不是"通信时间为 0"，而是"**通信不挡住计算**"——暴露时间趋近于 0。

## 5. 通信 Kernel 化：当 stream 重叠还不够

Stream 级重叠有个天花板：**NCCL kernel 和计算 kernel 是两个独立 kernel**，它们通过 event 依赖排序，边界处总有气泡；且 NCCL kernel 占多少 SM 是黑盒。

进阶路线是**把通信写进计算 kernel**——同一个 kernel 里，一部分 warp 做 GEMM，一部分 warp 做 put/get（这正是 15.4 NVSHMEM 的 device 侧能力提供的武器）：

| 代表 | 做法 | 收益 |
|---|---|---|
| **DeepEP** | MoE dispatch/combine 写成融合 kernel，warp 专职化（Part 2 的 Warp Specialization 思想复用） | 通信与 expert GEMM 在 kernel 内精细交错 |
| **Flux / 分布式 GEMM** | GEMM tile 算完一片立刻发一片（tile 粒度流水线） | 消除 kernel 边界气泡 |
| **TMA Multicast 思路** | Hopper 的 TMA 组播（Part 5 §5）本身就是"硬件级的一次写多处" | 节点内广播场景的极致形态 |

**思想内核与卷一卷二完全同构**：Warp Specialization（Part 2）、异步流水线（Part 5）、单 kernel 融合（Part 11 的 Kernel Fusion）——只是把舞台从"一颗 SM"扩大到了"一个集群"。这就是全书反复强调的：**通信优化不是新学问，是你已经会的微架构技巧在新尺度上的重演。**

## 6. 决策清单

拿到一个分布式训练任务，按顺序问：

1. **哪些通信不在关键路径？**（DP 梯度 → 分桶重叠；FSDP 参数 → prefetch）
2. **关键路径上的通信能否搬到快车道？**（TP 放 NVLink 域；开 NVLS）
3. **能否用无关计算填缝？**（MoE 用交错 micro-batch）
4. **还不够？才考虑 kernel 化**（NVSHMEM/融合通信 kernel）——复杂度陡增，先确认 1~3 已榨干。

## 7. 参考资料

- PyTorch DDP 设计文档（gradient bucketing）；FSDP 论文 *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*
- [DeepEP](https://github.com/deepseek-ai/DeepEP) 与 DeepSeek-V3 技术报告（DualPipe 调度）
- Megatron-LM / DeepSpeed-Ulysses 的 TP/SP 通信重叠实现
- Part 16（各并行策略的通信模式全景）、Part 17.3（c10d/FSDP 源码）
