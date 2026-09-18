# 15.3 NCCL 内部：LL / LL128 / Simple 协议、Channel 与拓扑感知

> 核心问题：NCCL 如何把通信算法映射到 NVLink + RDMA 的混合拓扑上？
>
> 前置：15.1（算子）、15.2（算法）。读完这一篇，你应该能描述一次 `ncclAllReduce` 从 Python 调用到 GPU 显存间数据落地的完整旅程，并知道 `NCCL_ALGO` / `NCCL_PROTO` / `NCCL_NCHANNELS` 这些环境变量在拧什么旋钮。

## 0. NCCL 是什么、不是什么

**NCCL（NVIDIA Collective Communications Library）** 是 NVIDIA 的 GPU 集合通信库，地位相当于"多 GPU 世界的 cuBLAS"：上层框架（PyTorch `DistributedDataParallel`、Megatron、DeepSpeed）只调它的 API，它负责把语义落到具体硬件上。

几个容易误解的点：

- **NCCL 不是 CPU 库**：它的通信 kernel 跑在 GPU 的 SM 上，buffer 是显存。CPU 只负责"发起"（host-side API）和 bootstrap（交换地址、建连接）。
- **NCCL 不是网络协议栈**：它复用底层传输——节点内走 NVLink/PCIe 的 P2P 显存直写，跨节点走 RDMA 网卡（IB/RoCE），NCCL 做的是**在这些传输之上编排集合算法**。
- **NCCL 与 MPI 的关系**：MPI 是 CPU 时代的通用通信标准；NCCL 借鉴其算子语义，但为 GPU 重写。两者可以共存（MPI 管跨节点 CPU 控制面，NCCL 管 GPU 数据面）。

## 1. 一次 AllReduce 的完整旅程

```
PyTorch: dist.all_reduce(tensor)
   │
   ▼
ProcessGroupNCCL（c10d，见 Part 17.3）：选 stream、记 event
   │
   ▼
ncclAllReduce(comm)                      ← host 侧 API
   │
   ├─ 1. 查 communicator：拓扑（topo graph）、ring/tree 结构（建 comm 时已算好）
   ├─ 2. 选算法：消息大小 × 拓扑 → Ring / Tree / NVLS / CollNet
   ├─ 3. 选协议：消息大小 → LL / LL128 / Simple
   ├─ 4. 切块：按 channel 数把数据切条，每条 channel 一个"子环"
   ▼
launch 1 个 NCCL kernel（常驻 SM 上跑，直到通信完成）
   │
   ├─ 节点内：P2P 直写对端显存（load/remote store over NVLink）
   ├─ 跨节点：把数据交给网卡（RDMA write），或经 proxy 线程转发
   ▼
完成 → stream 上插 event → 后续计算 kernel 可依赖它排序
```

注意一个反直觉的事实：**NCCL 通信是由 GPU 上的 kernel 执行的一**（P2P 部分）。通信 kernel 会占用 SM——这就是为什么通信和计算"真重叠"时要考虑 SM 抢占（15.5 的核心矛盾）。

## 2. Bootstrap 与拓扑探测：建 comm 时发生了什么

`ncclCommInitRank`（或 PyTorch 建 ProcessGroup 时）做三件事：

1. **Bootstrap**：通过 TCP（或 MPI）交换每个 Rank 的 GPU 显存地址、网卡 GID、CUDA IPC handle 等——这是 CPU 控制面，一次性。
2. **拓扑探测（Topo Discovery）**：读取本机 `nvidia-smi topo -m` 级别的信息——GPU 之间是 NVLink 还是 PCIe？经过几个 NVSwitch？哪张网卡离哪张 GPU 最近（同 NUMA/同 PCIe switch）？跨节点再合并成全局图。
3. **Graph Search**：在拓扑图上搜索 Ring / Tree 结构，使得环的每条边尽量落在"胖链路"上；输出每条 channel 的 Rank 排序表。

> 实操：`NCCL_TOPO_DUMP_FILE=topo.xml` 可以导出 NCCL 看到的拓扑，排查"为什么我的 AllReduce 只有理论带宽的一半"时先看它。

## 3. Channel：把胖链路切成多条流水线

**一个 Channel = 一条独立的逻辑环（或树）+ 一组专用的通信资源（buffer、线程块）**。

- 大数据被切成 `nChannels` 条，每条 channel 走不同的环（边集合可能不同，以打满多条物理链路）；
- 类比：Part 5 的"多 stage 异步流水线"——单条环喂不饱全部链路带宽，就多开几条并行流水；
- `NCCL_NCHANNELS`（或老版本 `NCCL_NTHREADS` 间接影响）可调；默认 NCCL 自动按拓扑决定。

效果：8 卡 NVSwitch 机型上，单 channel 的环只能用一部分链路，多 channel 可以把全互联带宽吃满。

## 4. 三种传输协议：LL / LL128 / Simple

同一个 Ring，NCCL 还有三种"怎么搬字节"的协议，区别在**数据与完成标志（flag）的耦合方式**——本质是 15.2 的 α-β 权衡在微观层面的再现：

| 协议 | 机制 | 适合 | 直觉 |
|---|---|---|---|
| **LL**（Low Latency） | 每 8 字节数据附带 8 字节 flag，一起写；接收方轮询 flag 确认到达 | **小消息** | "写数据的同时把'到了'的标签贴进去"，省一次握手，α 最小；但有效带宽减半（一半流量是 flag） |
| **LL128** | 128 字节为单位，其中 120B 数据 + 8B flag，利用 128B 原子写 | **中小消息** | flag 开销摊薄到 1/16，延迟与带宽的折中点 |
| **Simple** | 数据直接 DMA 大块写，完成后单独发 flag/信用更新 | **大消息** | 有效带宽最高（几乎无 flag 开销），但多一轮确认，α 大 |

**记忆锚点：消息越大，flag 占比越能摊薄，协议越"重数据轻确认"。** NCCL 默认按消息大小自动切换（可用 `NCCL_PROTO` 强制指定做实验）。

## 5. 传输层：P2P / SHM / NET

NCCL 把"相邻两个 Rank 之间怎么传"抽象成 transport：

| Transport | 场景 | 机制 |
|---|---|---|
| **P2P** | 同机，GPU 间有 NVLink/PCIe P2P 能力 | 通信 kernel 直接 `st.global` 到对端显存（CUDA IPC / NVLink 映射），不经过 CPU、不经过主存 |
| **SHM** | 同机但 P2P 不可用（如虚拟化限制） | 经主机共享内存中转，慢，尽量避免 |
| **NET** | 跨节点 | RDMA 网卡：**GPUDirect RDMA** 让网卡直接读写显存（见 Part 14.3），发送侧 SM 把描述符交给网卡后即脱身 |

`NCCL_DEBUG=INFO` 启动日志里会打印每条 channel 每个 hop 用了哪种 transport——性能排障第一证据。

## 6. NVLS：NVSwitch 代劳的 AllReduce

Hopper/Blackwell 节点（H100/B200 + NVSwitch）上，NCCL 会优先启用 **NVLS（NVLink SHARP）**：

- 传统 Ring：8 卡 AllReduce 要在环上走 2×(8−1) = 14 步，SM 上的通信 kernel 逐步收、加、发；
- NVLS：所有 SM 把数据写进 NVSwitch 的归约单元，**交换芯片完成求和**，SM 再读回结果——流量和 SM 占用同时大降（15.2 §5）。

这就是"为什么 H100 节点内 AllReduce 带宽能逼近 NVLink 理论值"的原因。前提是：buffer 需用 `ncclMemAlloc` 分配（注册进 NVLS 组），PyTorch 侧对应 NCCL 2.17+ 的 window 注册。

## 7. 常用调优/排障环境变量

| 变量 | 作用 | 典型用法 |
|---|---|---|
| `NCCL_DEBUG=INFO` | 打印拓扑、transport、协议选择 | 排障第一步 |
| `NCCL_DEBUG_SUBSYS=INIT,GRAPH` | 只看建图过程 | 拓扑问题定位 |
| `NCCL_ALGO=Ring/Tree/NVLS` | 强制算法 | A/B 实验 |
| `NCCL_PROTO=LL/LL128/Simple` | 强制协议 | 小消息延迟调优 |
| `NCCL_NCHANNELS=n` | channel 数 | 带宽打不满时调 |
| `NCCL_IB_GID_INDEX` / `NCCL_SOCKET_IFNAME` | 选网卡/接口 | 多网卡机型必查 |
| `NCCL_P2P_DISABLE=1` | 关 P2P（退回 SHM） | 验证 P2P 是否有问题 |

## 8. 与全书主线的呼应

- **通信 kernel 占 SM** ↔ Part 2 的 Warp Scheduler、Part 11 的 Occupancy：NCCL kernel 与你的计算 kernel 抢同一批 SM，15.5 的重叠技巧全是在解这个资源竞争。
- **flag 轮询** ↔ Part 13 的内存序：LL 协议的 flag 机制本质是 `st.release` / `ld.acquire` 语义的跨 GPU 版本，作用域从 `.cta`/`.gpu` 扩展到了 `.sys`。
- **channel 切条流水** ↔ Part 5 的双缓冲/多 stage：同一个"用流水掩盖延迟"的思想，从 SM 内搬到了 GPU 间。

## 9. 参考资料

- [NCCL 官方文档（Usage / Environment Variables）](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- [NCCL 源码](https://github.com/NVIDIA/nccl)：`src/graph/`（拓扑与建环）、`src/transport/`（P2P/SHM/NET）、`src/device/`（通信 kernel 与 LL/LL128/Simple 原语）
- [NVLink SHARP / NVLS 介绍（Hopper 白皮与 NCCL 2.12+ release note）](https://docs.nvidia.com/deeplearning/nccl/release-notes/)
