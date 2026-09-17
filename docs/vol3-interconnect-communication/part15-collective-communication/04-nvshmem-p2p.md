# 15.4 NVSHMEM 与 P2P：GPU 侧的远端内存语义

> 核心问题：kernel 内部如何直接读写另一台机器的 GPU 显存？
>
> 前置：15.1~15.3。读完这一篇，你应该理解 NCCL 与 NVSHMEM 的范式差异（host 发起的集合 vs device 发起的单边通信），以及为什么 MoE 训练框架（DeepEP 等）选择后者。

## 0. NCCL 范式的边界

15.3 讲的 NCCL 有一个隐含前提：**通信由 Host 发起，以"整段 buffer"为单位**。

```
Host:  ncclAllReduce(buf, ...)   ← CPU 发起一次"大动作"
GPU:   NCCL kernel 执行           ← GPU 只是执行者
```

这个范式对"规则的、块状的、提前知道通信对象"的场景（DP 梯度同步）非常合适。但有一类需求它覆盖不了：

> **kernel 跑到一半，才知道这个数据该发给谁。**

典型例子就是 **MoE**（Part 16.4）：每个 token 经过一个"路由器（gate）"网络，才决定发给哪个专家——而专家在别的卡上。也就是说，**通信的对端地址是 kernel 运行期才计算出来的**。NCCL 的 host 发起模型做不了这件事（host 无法预知每个 token 的去向），只能退回"先算路由、再 host 发 AllToAll"的两段式，多一次同步、多一轮数据落地。

**NVSHMEM 就是为解决这类问题而生：把通信指令下放到 device（kernel 内部）。**

## 1. PGAS：一个必须理解的编程模型

NVSHMEM 的理论根基是 **PGAS（Partitioned Global Address Space，分区全局地址空间）**：

- 所有 GPU 的显存被拼成一个**全局地址空间**，每块显存有"属主（哪个 PE/rank）"；
- 每个 rank 既可以访问自己的本地显存（快，本地 load/store），也可以用统一 API 访问**远端** rank 的显存（慢，走 NVLink/RDMA）；
- 关键约束：只有用 `nvshmem_malloc` 分配的**对称堆（Symmetric Heap）**才进入这个全局空间——"对称"指所有 rank 在同一偏移处持有同名变量，因此远端地址 = 本地基址的平移，**不需要在运行时交换指针**。

```
rank 0 对称堆: [....数据....]  基址 0xA000
rank 1 对称堆: [....数据....]  基址 0xA000   ← 偏移相同
rank 2 对称堆: [....数据....]  基址 0xA000

rank 0 的 kernel 里：nvshmem_float_put(远端地址=0xA000+off, 数据, 目标=rank 1)
                     → 直接写进 rank 1 的显存
```

> 与 CUDA 的 `cudaMalloc` + NCCL 对比：NCCL 世界里，"我要发给 rank 1"意味着 host 发起一次 collective；NVSHMEM 世界里，"我要发给 rank 1"是 kernel 里的一条 `put`——**通信的粒度从"一次操作"细到"一条指令"**。

## 2. 单边通信：put / get / signal

NVSHMEM 的核心 API 是**单边（one-sided）**的——发起方独立完成，不需要对端配合调用：

| API | 语义 | 类比 |
|---|---|---|
| `nvshmem_put(dest, src, size, pe)` | 把本地 `src` 写进 rank `pe` 的 `dest` | "我把快递塞进你家邮箱" |
| `nvshmem_get(dest, src, size, pe)` | 从 rank `pe` 的 `src` 拉回本地 `dest` | "我去你家取快递" |
| `nvshmem_put_signal(...)` | put 同时在远端更新一个信号量 | "快递到了，按一下门铃" |
| `nvshmem_wait_until(sig, ...)` | 等待本地/远端信号量达到某值 | "听到门铃才拆快递" |
| `nvshmem_quiet()` / `fence()` | 保证之前的 put 已落地/可见 | 见 Part 13 的内存序概念 |

`put_signal + wait_until` 组合就是 MoE dispatch 的标准动作：数据写过去、门铃按下去，对端专家 kernel 轮询到信号后立刻开始算。

### 与 Part 13 的衔接

Part 13 讲的内存序（acquire/release/proxy fence）在 NVSHMEM 里全部升级一个作用域：

- 本地显存的 release/acquire 语义 → 扩展到 `.sys` 作用域（跨 GPU 可见）；
- `quiet` ≈ 跨 GPU 版的"等待所有异步操作完成"（类比 `cp.async.wait_group`、TMA 的 bulk-group）；
- 对端可见性规则与 TMA/mbarrier 的完成协议在概念上是同构的：**数据落地**与**标志置位**必须保证先后顺序。

## 3. IBGDA：kernel 直接驱动网卡

NVSHMEM 跨节点的高性能来自 **IBGDA（InfiniBand GPUDirect Async）**：

- 传统路径：GPU kernel → 通知 CPU 代理线程 → CPU 给网卡下发 RDMA 任务（doorbell）→ 网卡搬数。CPU 在中间是延迟和抖动的来源。
- IBGDA：把网卡的队列（QP/WQE）映射进 GPU 地址空间，**kernel 里的线程直接填写 RDMA 工作请求并按门铃**，CPU 完全不参与数据面。

效果：跨节点 put 的延迟从 ~10 µs 级降到 ~2 µs 级，且 P99 抖动大幅收敛——对 MoE 这种"每步训练两次 AllToAll、每次由几千个小 put 组成"的负载，这个差异是能用与不能用的分水岭。

> 硬件依赖链：IBGDA = GPUDirect RDMA（Part 14.3）+ 网卡支持 device doorbell + NVSHMEM 运行时封装。

## 4. NVSHMEM vs NCCL：什么时候用哪个

| 维度 | NCCL | NVSHMEM |
|---|---|---|
| 发起方 | **Host**（CPU 调用） | **Device**（kernel 内发起） |
| 范式 | 集合（集合体协同一个大操作） | 单边 + 对称堆（PGAS） |
| 粒度 | 整个 buffer | 一条 `put`（可到字节级） |
| 通信对象 | 建 comm 时确定 | **运行期才知道也行** |
| 与计算的融合 | 通信 kernel 与计算 kernel 分离（靠 stream 重叠，见 15.5） | **可以融进同一个 kernel**（算着算着顺手 put） |
| 典型用户 | DDP/FSDP/TP 的梯度与激活 | **MoE dispatch/combine（DeepEP）**、稀疏 embedding 交换、不规则图计算 |

一句话：**规则块状的集合通信用 NCCL；不规则、路由运行时决定的细粒度通信用 NVSHMEM。** 现代 MoE 训练栈（如 DeepSeek 的 DeepEP）正是用 NVSHMEM 风格的 device 侧 put 实现了"路由-通信-计算"的单 kernel 融合。

## 5. 全景对照：三种通信范式的层级

```
层级3  NVSHMEM device 侧：kernel 内一条 put   ← 最细粒度，运行期路由
层级2  NCCL kernel 侧：    一个 kernel 完成整个 AllReduce
层级1  Host API 侧：       ncclAllReduce(...) / dist.all_reduce(...)   ← 最粗粒度，最易用
```

这与卷二的软件栈层级（CUDA C++ → PTX → SASS）是同一种"越往下越灵活、越难写"的结构。初学者从层级 1 入手，理解 15.1/15.2 的语义与算法；做 MoE/稀疏方向再下沉到层级 3。

## 6. 参考资料

- [NVSHMEM 官方文档](https://docs.nvidia.com/hpc-sdk/nvshmem/api/)
- [DeepEP（DeepSeek 开源的 MoE 通信库）](https://github.com/deepseek-ai/DeepEP)：NVSHMEM/IBGDA 在 MoE dispatch-combine 的工业级实现，Part 16.4 与 Part 12 的源码分析对象之一
- PGAS 概念源流：UPC / OpenSHMEM 规范（NVSHMEM 是其 GPU 实现）
