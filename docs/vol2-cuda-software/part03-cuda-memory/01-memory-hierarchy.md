# 3.1 存储层次与真实数据路径

## 3.1.0 不要只背“延迟金字塔”

CUDA 存储层次真正影响 kernel 设计的是四个维度：

1. **谁拥有、谁可见**：thread、CTA、cluster、device 还是 system；
2. **数据放在哪里**：寄存器堆、SM SRAM、L2、HBM/GDDR、host DRAM；
3. **谁管理**：编译器、程序员、cache 硬件、驱动页迁移还是异步 copy engine；
4. **如何交接**：普通依赖、barrier、memory fence、async completion。

| 空间/资源 | 典型位置 | 可见范围 | 管理方式 | 常见瓶颈 |
|---|---|---|---|---|
| Register | SM register file | thread | 编译器 | pressure、spill、occupancy |
| Shared Memory | SM 上 SRAM | CTA；Hopper+ 可经 DSM 扩展到 cluster | 程序员 | 容量、bank conflict、同步 |
| Local Memory | 实际在 device memory，经 cache | thread 私有地址语义 | 编译器 spill/数组放置 | 高延迟、额外带宽 |
| L1/Data Cache | SM 上 SRAM，常与 shared 共享物理资源 | SM | 硬件 + carveout hint | thrashing、利用率低 |
| L2 | GPU 全局 cache | device | 硬件 + persisting/hint | 容量竞争 |
| Global Memory | HBM/GDDR | device | 程序员/allocator | latency、coalescing、带宽 |
| Constant | device memory + constant cache | device，只读 | 程序员 | warp 地址分歧时串行 |
| Texture | device memory + texture path | device，只读 | texture object | 不适合普通写路径 |
| Unified Memory | 统一虚拟地址，物理页可迁移 | system 语义 | driver/runtime | page fault、迁移抖动 |
| Pinned Host Memory | host DRAM 锁页 | host/device DMA 可达 | runtime | 分配昂贵、过多会伤害系统 |
| TMEM | Blackwell DC Tensor 专属片上空间 | 指令规定的 CTA group | 显式 alloc/dealloc | 严格生命周期与同步 |

延迟数字会随芯片、命中层级和访问模式变化。优化时优先问“流量经过了哪些路径、能否重用、是否并行”，不要拿一张固定周期表替代实测。

## 3.1.1 Global Memory 与 3.1.7 L2/L1 Cache

`cudaMalloc` 得到的内存物理上位于 HBM/GDDR，普通 load/store 通常先经过 L2，再按指令 cache policy 决定 L1 行为。三个容易误判的点：

- “显存带宽很高”不代表 kernel 不缺带宽；Tensor Core 峰值增长更快。
- “请求连续”只是第一步，还需看请求利用率、并发 outstanding 数量和 L2 命中。
- cache policy 是提示或局部控制，不会修复错误的数据布局。

Ampere 起可为反复访问的窗口配置 L2 persisting policy。它适合尺寸可控、复用稳定的热点，不适合把整个工作集都标成 persist。

## 3.1.2 Shared Memory：软件管理的片上 staging area

Shared Memory 的核心价值不是“比 global 快”，而是**将一次片外加载转化为 CTA 内多次片上复用**。典型 GEMM：

```text
HBM 中 A/B tile
  → global load / cp.async / TMA
  → shared memory 中按 Tensor Core 消费方式重排的 tile
  → ldmatrix / WGMMA / tcgen05 消费
```

Shared Memory 与 L1 在很多架构上共享物理 SRAM，但地址空间、可见性和管理方式不同。`cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, ...)` 只是偏好；静态/动态 shared 容量、每 SM 上限和 occupancy 仍要按目标芯片核对。

### 生命周期

- CTA 启动后，其 shared storage 才存在；
- 访问 remote DSM 前，目标 CTA 必须已完成初始化；
- CTA 退出前，cluster 中其它 CTA 不能再访问它的 shared storage；
- 异步 copy 的目标 buffer 不能在 completion 前消费或覆盖。

## 3.1.3 Register 与 3.1.6 Local Memory

寄存器通常是最低延迟的线程私有存储，但“多用寄存器一定更快”是错的：

- 每线程寄存器增多会减少同驻 warp/CTA；
- accumulator、地址、pipeline state 会同时争夺 register file；
- 超过分配能力会 spill 到 local memory；
- local memory 有线程私有的地址语义，却位于片外内存路径。

用 `nvcc -Xptxas=-v` 查看 register 和 spill，再用 Nsight Compute 判断 occupancy 与 stall。不要为了追求高 occupancy 盲目 `--maxrregcount`，它可能制造更多 spill。

## 3.1.4 Constant Memory 与 3.1.5 Texture Memory

Constant cache 对 warp 内相同地址具有广播优势；若 32 个 lane 读取 32 个不同 constant 地址，请求可能被序列化。适合小型只读参数、卷积系数、查表常量。

Texture path 适合空间局部性、插值与边界模式。现代只读普通数据不应机械地套用旧教程中的 `__ldg()`；编译器、cache 层次与 API 已演进，应根据目标架构生成代码和 profile 决定。

## 3.1.8 Unified Memory

`cudaMallocManaged` 提供统一虚拟地址，不意味着 CPU/GPU 同时无代价地访问同一物理副本。常见过程是：

```text
首次 GPU 访问
  → page fault / access counter
  → 驱动迁移或建立远程映射
  → TLB 更新
  → kernel 继续
```

可通过 `cudaMemPrefetchAsync`、`cudaMemAdvise`、访问阶段划分降低 fault 和 ping-pong。Unified Memory 的目标首先是可编程性；规则访问的大吞吐路径往往仍适合显式管理。

## 3.1.9 Pinned、Mapped 与异步 H2D/D2H

DMA 期间物理页不能被操作系统换出，因此 pageable host memory 往往需要 staging；`cudaMallocHost`/`cudaHostAlloc` 的 pinned memory 可直接参与 DMA，并支持真正可重叠的异步传输路径。

注意：

- pinned allocation/registration 成本高，不应高频创建销毁；
- 锁页太多会压缩系统可分页内存；
- `cudaMemcpyAsync` 能否与 kernel 重叠还取决于 stream、copy engine、依赖和设备能力；
- mapped zero-copy 省去显式复制，但离散 GPU 直接访问 host memory 的延迟/带宽通常远差于 device memory。

## 3.1.10 从“空间”转向“数据通路”

后续章节统一用下面五问分析任意搬运：

1. source/destination 是什么地址空间？
2. 请求由多少线程发起，地址由谁计算？
3. 数据是否经过寄存器？
4. 完成由普通依赖、async group 还是 mbarrier 表达？
5. buffer 在何时可被消费者读取、复用或释放？

下一篇：[访问模式、transaction 与 swizzle](02-access-patterns.md)。
