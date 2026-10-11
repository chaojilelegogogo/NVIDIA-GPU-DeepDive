# 6.6 Part 2.5：Warp Specialization —— 把流水线角色分给不同的 Warp

> 核心问题：Hopper 上 `setmaxnreg` 到底解决什么问题、约束是什么？TMA/WGMMA 的 handoff 协议怎么写才不死锁？FlashAttention-3 和 CUTLASS 的角色是怎么分的？
>
> 前置：[6.5 Hopper：TMA、`mbarrier` 与 `wgmma.mma_async`](05-hopper-wgmma.md)、[第二部分 2.8 Producer-Consumer 模式](../../vol1-gpu-hardware/part02-execution-pipeline.md)。同步原语的完整语义配合[第十三部分](../../vol2-cuda-software/part13-synchronization-handbook/03-async-pipelines.md)阅读。
>
> 本篇是第六部分里唯一"以编程范式而非指令"为主角的章节——因为 Warp Specialization 不是一个指令、也不是一个 API，而是**一组硬件特性（TMA + WGMMA + `mbarrier` + `setmaxnreg`）组合出来的一种 kernel 组织结构**。

## 6.6.1 为什么需要它：把 Ampere 的流水线拆开看

[第二部分 2.8](../../vol1-gpu-hardware/part02-execution-pipeline.md) 讲过 Producer/Consumer 模式：一个 kernel 里，一边发起下一批数据的搬运、一边用当前已就绪的数据计算，两者时间重叠。Ampere 上这个模式的实现方式是：

```text
// Ampere 风格：所有 warp 做同样的事
for (int k = 0; k < K_TILES; ++k) {
    cp.async(dst = smem[buf][k+1], src = gmem[k+1]);   // 每个线程各自发起
    cp.async.commit_group();
    cp.async.wait_group<1>();                          // 等上一批
    __syncthreads();                                   // 全 CTA 会合
    mma_sync(acc, smem[buf][k]);                       // 每个 warp 各算自己那份
}
```

这个结构正确且有效，但在 Hopper 上有三处结构性浪费——**注意它们都不是"代码写得不够好"，而是这个组织方式本身的固有代价**：

| # | 问题 | 具体表现 |
|---|---|---|
| 1 | **发起与等待耦合在同一个 warp 里** | 同一个 warp 既要发起搬运/MMA、又要等它完成，等待期间这个 warp 什么都做不了。虽然调度器可以切到别的 warp，但"发起者"和"消费者"是同一批 warp 时，wait 的 latency 只能靠"更多 warp"来填，而 warp 数受寄存器限制 |
| 2 | **寄存器配额被迫均分** | 所有 warp 用同一个 `.maxnreg`。但事实上：搬运只需要几个寄存器（地址、循环变量），计算需要几百个（累加器、softmax 中间量）。均分的结果是**计算 warp 不够用、搬运 warp 用不完**，前者被逼出 register spilling 或只能缩小 tile |
| 3 | **非矩阵计算与 MMA 抢同一个发射端口** | GEMM 的 epilogue、Attention 的 softmax/量化都是标量运算。它们和 `mma.sync`/`wgmma` 由同一批 warp 发出，就与矩阵指令竞争 warp scheduler 的发射带宽 |

第 2 点最直观：Hopper 把 FP32 通路翻倍到 128/SM（见 H100 手册 2.1），单线程寄存器上限是 255，但**这是个"上限"不是"分配"**——如果不做点什么，编译器只能按所有 warp 的共同需求均分。

> Warp Specialization 的思路一句话：**既然不同工作的资源画像完全不同，就让不同的 warp 专职做不同的事，并把寄存器按需重新分配。**

## 6.6.2 角色划分：三种典型 Warp Group

Warp Specialization 在实践中收敛出三种角色（以 Attention/GEMM 类 kernel 为例）：

| 角色 | 职责 | 典型规模 | 主要指令 |
|---|---|---|---|
| **Producer**（生产者） | 计算 tile 坐标、发起 TMA、用 `expect_tx` 登记搬运量、维护 ring buffer 索引 | 1 个 Warp Group（4 warps / 128 线程） | `cp.async.bulk.tensor.*`、`mbarrier.arrive.expect_tx` |
| **Consumer**（消费者） | 等待数据就绪、发起 WGMMA、做非矩阵运算（softmax、缩放） | 1~2 个 Warp Group | `wgmma.mma_async`、`wgmma.commit_group`/`wait_group`、标量运算 |
| **Epilogue**（收尾） | 等累加器就绪、写回 Global Memory | 1 个 Warp Group（可与 Consumer 合并） | `st.global`/TMA store、`setmaxnreg` |

时间轴（单 stage、两角色）：

```text
Producer WG:  [TMA tile0] [TMA tile1] [TMA tile2] [TMA tile3] ...
Consumer WG:              [WGMMA tile0] [WGMMA tile1] [WGMMA tile2] ...
                              ↑ 等 full barrier        ↑ 同时 softmax 与 tile1 的 WGMMA 重叠
```

**重要澄清（一个常见误解）**：Warp Specialization 不是 CUDA 语言特性，**没有任何语法糖**。"哪个 warp 当什么角色"是你用 `if (warp_id < 4) {...} else {...}` 自己分的；`mbarrier` 也不会因为你分了角色就自动握手——**角色划分只提供服务，不提供同步**。这一点在[第十三部分 13.3](../part13-synchronization-handbook/03-async-pipelines.md) 有专门强调。

## 6.6.3 关键使能：`setmaxnreg` 动态寄存器重分配

### 为什么需要它

Hopper 之前，一个 kernel 的寄存器预算是**静态**的：编译期决定 `.maxnreg`（或由 `__launch_bounds__` 推出），运行期所有线程按同一配额分配。上面问题 2 就来自这里。

Hopper（`sm_90a`）新增了 **`setmaxnreg`** 指令：**在 kernel 运行期，按 Warp Group 为单位动态调整寄存器配额**，在同一 CTA 内部重新分配这块固定的寄存器总量（寄存器堆 256 KB/SM 这块"十年不涨"的资源，见 A100 手册的讨论——正因为总量不变，"重分配"成了唯一出路）。

### PTX 语法

```ptx
// 增加本 Warp Group 的寄存器配额（最多能用到 imm 个/线程）
setmaxnreg.inc.sync.aligned.u32 232;

// 减少本 Warp Group 的配额（把额度让出去）
setmaxnreg.dec.sync.aligned.u32 40;
```

典型用法是**生产者让、消费者拿**：

```cuda
if (is_producer_warpgroup) {
    // 搬运只需要少量寄存器：地址、循环变量、barrier 状态
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    producer_loop();
} else {
    // 计算需要大量寄存器：累加器、softmax 统计量、operand fragment
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    consumer_loop();
}
```

CUTLASS 里对应 `cute::setmaxnreg_dec<>` / `cute::setmaxnreg_inc<>` 封装；CuTe 的 `CollectiveBuilder` 会自动按 warp specialization policy 生成这两条指令。

### 语义与约束（这五条是写代码前必须知道的）

1. **以 Warp Group 为单位，不是以 Warp 为单位**：立即数是"每个线程最多能用到多少个寄存器"，但**语义上是对整个 128 线程的 Warp Group 生效**，硬件按 Warp Group 粒度跟踪。
2. **必须由同一 Warp Group 的全部线程集体执行**，且立即数相同（`.sync.aligned`）。写在分支里时，要保证该 Warp Group 内不会分歧到只有一个线程执行它。
3. **立即数必须是 8 的倍数**。
4. **总量守恒，不能凭空造寄存器**：`setmaxnreg` 只能在 CTA 启动时申请到的寄存器总量内重新分配。因此**编译期的寄存器上限必须按"最高需求者"设定**——通常做法是用 `__launch_bounds__`（或 `-maxrregcount`）把上限设到消费者需要的值（如 232），再由生产者主动 `dec` 让出。
5. **只存在于 `sm_90a`**（Hopper 数据中心芯片）。消费级 Blackwell（`sm_120`）和 Ampere 上都没有这条指令，写了会编译失败。

### 配额算术：一个具体例子

假设 CTA = 1 个 Producer Warp Group（128 线程）+ 2 个 Consumer Warp Group（256 线程）＝ 384 线程，SM 寄存器堆 65536 个（256 KB ÷ 4 B）：

\[
\text{总量} = 384\ \text{线程} \times \text{平均配额}
\;\Rightarrow\; 65536 / 384 \approx 170\ \text{寄存器/线程（均分时）}
\]

若不重分配，所有人都只有 ~170。重分配后：

- Consumer：`inc` 到 **232**，两个 Consumer WG 共占 \(256 \times 232 = 59392\)
- 剩余 \(65536 - 59392 = 6144\)，分给 Producer 的 128 线程 \(\Rightarrow\) 每个线程 \(6144/128 = 48\)

于是 Producer `dec` 到 **40~48** 即可稳定成立。**这个"消费者 232、生产者 40 左右"就是 FlashAttention-3 / CUTLASS Hopper kernel 里最常见的配额组合**（具体值随 kernel 与 toolkit 版本略有差异，以实测为准）。

> 与 Occupancy 的关系（呼应术语表中 Occupancy 词条）：`setmaxnreg` 是在**同一个 CTA 内部**重分配，不改变 CTA 占用的寄存器总量，因此**它不直接提高 Occupancy**；它提高的是"在给定 Occupancy 下，计算 warp 能持有的状态量"——从而让累加器更大、softmax 中间量不落 local memory，间接减少 stall。

### 如何验证生效

`setmaxnreg` 会体现在 SASS 里（Hopper SASS 有对应的 `SETMAXNREG` 指令形式），并且会改变 `cuobjdump` 显示的寄存器用量分布。验证方法：

```bash
# 1. 看寄存器用量与 spill（编译期视角）
nvcc -arch=sm_90a --ptxas-options=-v kernel.cu

# 2. 反汇编确认 setmaxnreg 出现在正确的位置（运行期视角）
cuobjdump -sass kernel.cubin | grep -i setmaxnreg

# 3. Nsight Compute 看实际驻留寄存器与 stall 归因
ncu --set full ./kernel          # 关注 Registers Per Thread 与 Warp State
```

> 注意：`setmaxnreg` 是**运行期**生效的，所以 `--ptxas-options=-v` 打印的是"编译期分配的上限"，而不是最终的动态配额。两者都要看。

## 6.6.4 握手协议：`mbarrier` 的 full / empty 双屏障

角色分好之后，最大的坑是**同步**。生产者要告诉消费者"数据到了"，消费者要告诉生产者"缓冲区用完了"，而 `__syncthreads()` 在这里是错的——它是全 CTA 的屏障，会把"生产者提前跑下一轮"这个最大收益直接抹掉。

Hopper 的答案是 **`mbarrier`**（见 [6.5.2](05-hopper-wgmma.md) 与第十三部分 13.3），用两个屏障构成 ring buffer 的握手：

```text
Producer:  ──arrive(expect_tx=N)──▶ [ full[i] ]  ──▶  Consumer: wait(full[i])
Consumer:  ──arrive()────────────▶ [ empty[i] ] ──▶  Producer: wait(empty[i])
```

- **full barrier**：生产者用 `mbarrier.arrive.expect_tx` 登记"我这次要搬 N 字节"，然后**立刻返回继续搬下一块**（不等！）；TMA 完成后硬件自动递减 transaction 计数，计数归零时屏障 phase 翻转，消费者的 `try_wait/try_wait_parity` 才通过。
- **empty barrier**：消费者用完这个 stage 后 `arrive`，生产者下一轮在复用这个 stage 前先 `wait(empty[i])`。

**这里就是 Warp Specialization 与普通流水线最关键的区别**：producer 的 wait 发生在"下一轮复用同一 stage 之前"（提前很远），而不是"这一轮搬运之后"（立刻等待）。所以生产者几乎从不阻塞。

一个最小的双角色骨架（省略坐标计算与 descriptor 细节）：

```cuda
__global__ void ws_gemm(...) {
    __shared__ barrier full[STAGES], empty[STAGES];
    __shared__ alignas(128) float smem[STAGES][TILE_BYTES];

    if (threadIdx.x < PRODUCER_THREADS) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (elect_leader()) {                      // 只需一个线程发 TMA
            for (int s = 0; s < STAGES; ++s) init(&empty[s]);
        }
        for (int k = 0; k < K_TILES; ++k) {
            int s = k % STAGES;
            wait(&empty[s]);                        // 复用前先等消费者腾空
            if (elect_leader()) {
                expect_tx(&full[s], TILE_BYTES);
                tma_load(smem[s], tensor_map, coords(k), &full[s]);
            }
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        accum_t acc = {};                           // 大量寄存器在这里
        for (int k = 0; k < K_TILES; ++k) {
            int s = k % STAGES;
            wait(&full[s]);                         // 数据就绪
            wgmma(acc, smem[s]);                    // 从 Shared descriptor 读操作数
            wgmma_commit_group();
            if (elect_leader()) arrive(&empty[s]);  // 声明"这个 stage 我用完了"
        }
        wgmma_wait_group<0>();
        epilogue(acc);
    }
}
```

### 四个最容易踩的坑

| 坑 | 现象 | 根因与对策 |
|---|---|---|
| **死锁** | kernel 挂住不结束 | barrier 的 arrive 计数（`init` 时的 count）与实际参与线程数不一致；或漏了一次 `arrive`。**用 `elect.sync` 选出单一 leader 来 arrive，而不是靠 `threadIdx.x == 0`**（见[第十三部分 13.5.2](../part13-synchronization-handbook/05-collectives-and-atomics.md)） |
| **phase 用错** | 偶发读取到未就绪数据 | `try_wait_parity` 的 parity 参数需要手动翻转；用 `cuda::barrier` 封装可以避免 |
| **stage 数不足** | 生产者被 `empty` 挡住，重叠失效 | STAGES 太少 → 生产者没 space 提前跑；但 STAGES 太多会吃掉 Shared Memory（挤压 Occupancy）——正是[第二部分 2.9](../../vol1-gpu-hardware/part02-execution-pipeline.md) 讲的 tradeoff |
| **`setmaxnreg` 位置错** | 编译失败或行为未定义 | 必须在该 Warp Group 进入自己的循环**之前**、由全体线程集体执行；不能放在只有部分线程到达的分支里 |

> 一句话总结协议：**full 管"数据能不能读"，empty 管"缓冲区能不能写"，两个屏障的 wait 位置决定了重叠的上限。**

## 6.6.5 与 Ampere 写法的正面对照

| 维度 | Ampere（统一角色） | Hopper（Warp Specialization） |
|---|---|---|
| 谁发起搬运 | 每个线程各发一小块 `cp.async` | **一个** Producer WG，单线程发 TMA（描述符驱动） |
| 谁发起计算 | 每个 warp 各自 `mma.sync` | Consumer WG 集体 `wgmma.mma_async` |
| 寄存器分配 | 静态均分 | **动态**：`setmaxnreg` 让生产者让给消费者 |
| 同步原语 | `cp.async.wait_group` + `__syncthreads()` | `mbarrier` full/empty（**不用 CTA 级屏障**） |
| 搬运与计算的解耦度 | 中等（同一个 warp 既发又等） | 高（不同 warp，且 wait 位置被推后一整个 stage） |
| 代码复杂度 | 低 | **高**（要自己管 barrier 状态机与角色） |
| 适用场景 | 大部分 kernel | 只有"搬运 + 矩阵计算"占比极高、且值得为此付出复杂度的 kernel |

**最后一行尤其重要：不要无脑上 Warp Specialization。** 对 Element-wise、Reduce、小型 GEMM 这类 kernel，角色划分带来的收益远不足以抵消复杂度与调试成本。判断标准：**这个 kernel 的 Tensor Core 利用率是否被"非矩阵工作"挡住了？** 如果是（典型如 Attention、大 GEMM），才值得。

## 6.6.6 工业级实现：FlashAttention-3 与 CUTLASS

### FlashAttention-3 的角色划分

[第十二部分 12.8](../part12-kernel-source-analysis.md) 提到了 FA3"全面转向 Warp Specialization"，这里补上它具体是怎么分的：

- **1 个 Producer Warp Group**：专职发起 K/V 分块的 TMA 加载；
- **2 个 Consumer Warp Group（ping-pong）**：一组算 `QK^T` 与 softmax，另一组算 `×V` 的 WGMMA，两组交替推进——这就是第二部分 2.9 节提到的"两个 Warp Group 交替扮演角色"的 Ping-Pong 模式；
- **Softmax 与 WGMMA 的重叠**：这是 FA3 的核心收益点。softmax 是低吞吐标量运算，传统写法里它会在关键路径上挡住 Tensor Core；拆出角色后，softmax 在一个 Consumer WG 里做的同时，另一个 Consumer WG 的 WGMMA 正在跑——**用"异步指令 + 角色分离"把标量延迟藏进矩阵计算的空隙**；
- 角色分离还顺带让 FA3 可以用更激进的寄存器预算（消费者 232），把 accumulator 与 softmax 中间量都留在寄存器里。

### CUTLASS 的 Warp Specialization Policy

CUTLASS 3.x 把角色划分**模板化**了：通过 `CollectiveBuilder` 指定 `num_mma_warps`（消费者）、`num_load_warps`（生产者）、以及是否启用 `setmaxnreg`，由库生成对应的 barrier 状态机与寄存器重分配代码。这是"理解原理后用库"的正确姿势：

- **读源码**：看 `sm90/collective/` 下 warp-specialized 的 Collective 实现，对照本篇 6.6.4 的骨架——注意 CUTLASS 的封装层次（`CollectiveBuilder` → `CollectiveMma` → 底层 `setmaxnreg` 与 barrier 代码）比骨架多出的是工程健壮性，不是新的硬件机制；
- **自己写**：先用本篇的骨架跑通一个最小 WS GEMM，再对照 CUTLASS 看它多处理了哪些边界（多 stage 的 barrier 数组、predicated TMA、epilogue 独立 WG）。

## 6.6.7 Evolution：Blackwell 上角色进一步解耦

Warp Specialization 在 Blackwell 上继续沿着"解耦"方向走（细节见 6.7 与第二部分 2.10）：

- **发起颗粒度从 Warp Group 放松到单线程**：某些 `tcgen05.mma` 变体单线程即可发起（结果落 TMEM，不再受"发起者 = 结果拥有者"约束），角色划分的自由度更大；
- **`.ws` 变体**：`tcgen05.mma.ws` 明确为 Warp Specialization 场景设计；
- **角色进一步专精**：TMEM 的引入让"累加器"从消费者 warp 的寄存器里搬到专属存储，消费者 warp 的寄存器压力显著下降——**这反过来削弱了 `setmaxnreg` 的必要性**（因为它存在的理由之一就是"寄存器不够分"）。

这条演进链的规律和全书一致：**每一代把"上一代需要软件小心编排的东西"变成硬件默认行为**——但代价是硬件形态更专用（`sm_120` 消费级就没有 TMEM）。

## 6.6.8 与全书主线的呼应

- **搬运主线**：Warp Specialization 是"搬运职责从计算单元剥离"这条线在**软件组织结构**上的体现——TMA 把地址生成从 warp 手里拿走，WS 则让"发起搬运"彻底不再是计算的副业。
- **协作范围主线**：本节的协作颗粒度是 Warp Group（128 线程），正是 Hopper 新引入的那一层（[第二部分 2.8](../../vol1-gpu-hardware/part02-execution-pipeline.md)、[第四部分 4.10](../part04-cuda-programming-model.md)）。
- **资源主线**：`setmaxnreg` 是"寄存器堆十年不涨"（[第一部分 1.4](../../vol1-gpu-hardware/part01-gpu-hardware-architecture.md)、A100 手册）逼出来的又一个软件侧补救——与 TMEM（Blackwell）是同一问题的两种回答：一个重分配，一个搬出去。
- 下游：[第十一部分 11.4](../part11-performance-optimization.md)（作为流水线优化手段）、[第十二部分 12.8](../part12-kernel-source-analysis.md)（FA3/FlashMLA 的实现分析）。

## 6.6.9 参考资料

- [CUDA C++ Programming Guide：`setmaxnreg` 与 Warp Specialization 相关章节](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（随 Toolkit 版本更新，务必核对你使用的版本）
- PTX ISA：`setmaxnreg`、`mbarrier`、`cp.async.bulk.tensor`、`wgmma.mma_async` 各章节
- [CUTLASS 文档：Hopper Warp Specialization 与 `cute::setmaxnreg`](https://github.com/NVIDIA/cutlass)
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608)（角色划分与 ping-pong 调度的原始描述）
- [Colfax Research：CUTLASS Tutorial / WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [第十三部分 13.3 异步流水线](../part13-synchronization-handbook/03-async-pipelines.md)、[13.5 `elect.sync` 与 leader 选举](../part13-synchronization-handbook/05-collectives-and-atomics.md)

下一篇：[Blackwell TMEM 和 `tcgen05`](07-blackwell-tcgen05.md)。
