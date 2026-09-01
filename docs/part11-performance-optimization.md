# 第十一部分 CUDA 性能优化（实战）

> 前面十部分建立的是"为什么"的知识体系。这一部分把它们整理成一份可以直接在实战中使用的检查清单和方法论。每一条优化手段都会标注它对应第几部分的哪个硬件机制——**优化不是背口诀，而是针对具体硬件瓶颈的对症下药**。

## 11.1 优化方法论：先诊断，后开药

正确的优化顺序永远是：

```
1. 用 Roofline 模型（Part 1.6）判断 kernel 是 Compute Bound 还是 Memory Bound
2. 用 Nsight Compute 的 Warp State/Stall 分析（Part 9.7）定位具体瓶颈类型
3. 针对定位到的瓶颈，选择对应的优化手段（本部分下面按类别展开）
4. 修改后重新测量，确认瓶颈是否真的转移/消除，而不是主观猜测
```

最常见的错误是跳过 1、2 步直接开始"优化"——比如对一个已经是 Compute Bound 的 kernel 拼命做访存合并，或者对一个已经 Memory Bound 到极限的 kernel 死磕指令级并行，这些努力都不会带来实质提升。

## 11.2 Memory 优化

对照第三部分：

- **Coalescing**：确保 Warp 内线程访问的 Global Memory 地址连续且对齐；结构体数组考虑 AoS→SoA 转换。
- **Vector Load**：用 `float4`/`int4` 等向量类型一次搬运 16 字节（对应 SASS 的 `LDG.128`），减少指令数、更好地利用一次内存事务的带宽，前提是数据本身对齐且访问模式允许合并成向量宽度。
- **Bank Conflict**：Shared Memory 二维数组访问前检查 stride 是否等于（或是）Bank 数量的整数倍；用 Padding（`tile[N][M+1]`）或 Swizzle 打散冲突。
- **Prefetch**：显式预取——在真正需要数据之前提前发起加载（可以是简单的提前几行读入寄存器，也可以是用 `cp.async`/TMA 构建软件流水线，见 11.4）；也可以用 `prefetch.global.L2` 之类的指令给 Cache 替换策略提示。
- **Cache 策略选择**：`__ldg()`/`const __restrict__` 引导只读数据走独立的只读数据路径；对一次性使用的数据用 `.cg`/`.cs` 修饰符避免污染 L1。
- **精度/数据类型选择**：能用 FP16/BF16/FP8/INT8 就不用 FP32——这不仅减少计算量，更直接减少了访存的字节数，往往是 Memory Bound kernel 最有效的单一优化手段（本质上是在提高 Part 1.6 的 Arithmetic Intensity）。

## 11.3 Compute 优化

对照第四、五部分：

- **减少指令数**：用 `-use_fast_math`（在允许精度损失时）、`__expf`/`__logf` 等快速数学函数走 SFU 而不是软件展开的多指令实现。
- **Loop Unrolling（循环展开）**：`#pragma unroll` 显式展开循环，减少循环控制开销（`IADD`/`ISETP`/`BRA`），并给编译器更大的指令调度空间去构造指令级并行、填充延迟（对照第九部分 9.6）。展开也不是越多越好——过度展开会增加寄存器压力（挤压 Occupancy）、增加代码体积（可能影响指令 Cache 命中），需要实测权衡。
- **善用 Tensor Core**：能表达成矩阵乘法的计算尽量转化为 GEMM 形式交给 Tensor Core（第五部分），标量 ALU 路径几乎完全无法与专用矩阵乘法电路的吞吐相提并论。
- **Dual Issue 友好的指令混合**：避免整段代码全是同一类型指令（比如全是访存或全是浮点运算），适当的指令类型混合有助于编译器利用不同执行单元的并行发射能力。

## 11.4 Pipeline 优化

对照第三、四部分：

- **Double/Triple Buffer**：用 `cuda::pipeline`（第七部分 7.5）或手写 `cp.async`/`mbarrier` 序列，构建搬运与计算重叠的软件流水线。级数（stage 数）的选择需要在"更深流水线掩盖更多延迟波动"和"更多 Shared Memory 占用挤压 Occupancy"之间权衡，通常通过实测在 2/3/4 级之间选择最优点。
- **Warp Specialization**（Hopper 起，第四部分 4.8）：把生产者（TMA 搬运）和消费者（WGMMA 计算）分给不同 Warp Group，用 `setmaxnreg` 精细调配寄存器配额。
- **减少同步粒度**：优先用 `__syncwarp()` 而不是 `__syncthreads()`（如果逻辑上只需要 Warp 内同步），减少不必要的等待范围（对照第二部分 2.10 的同步代价分层）。

## 11.5 Occupancy 优化

对照第二、九部分：

- 用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 给编译器提供寄存器分配的目标提示。
- 用 CUDA Occupancy Calculator / `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API 在编译期/运行时评估不同 Block 大小下的理论 Occupancy。
- **牢记 Occupancy 不是目标，是手段**：只有当 Nsight Compute 显示瓶颈是"延迟没有被充分隐藏"（如 `stall_long_scoreboard` 占比高，同时 Occupancy 明显偏低）时，提高 Occupancy 才是对症的药；如果 kernel 已经是 Compute Bound 且指令级并行度良好，过度压低寄存器数换 Occupancy 反而可能因为寄存器溢出（Local Memory 访问，第三部分 3.1.6）而变慢。

## 11.6 Register 优化

- 减少每线程使用的中间变量、及时让不再需要的变量"死亡"（缩小生命周期，帮助编译器复用寄存器）。
- 警惕大数组的局部变量（容易被编译器放入 Local Memory 而非寄存器，尤其是下标非编译期常量的数组）。
- 用 `--ptxas-options=-v` 编译查看每个 kernel 实际使用的寄存器数、Spill Store/Load 次数，作为寄存器压力的直接证据。

## 11.7 Persistent Kernel（常驻内核）

- **What**：不让 Kernel 按传统方式启动一次处理完所有数据就退出，而是启动**固定数量、刚好等于 GPU 实际能同时驻留的 Block 数**的 Kernel，让每个 Block（通常配合 `%smid`，第六部分 6.6）在内部用循环反复从一个工作队列里领取新的任务，直到全部任务处理完毕才真正退出。
- **Why**：避免了反复启动 Kernel 带来的 Launch Overhead（第七部分 7.3），也避免了每次启动新一批 Block 时，GigaThread Engine 重新做 Block 到 SM 分发决策的开销；对于任务粒度很细、数量巨大、且需要更精细的负载均衡（比如工作窃取 Work-Stealing）的场景尤其有效。
- **代价**：编程复杂度显著提高，需要手工实现任务队列和负载均衡逻辑；且要求 Kernel 启动时精确匹配硬件能同时驻留的 Block 数（常配合 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算），如果启动的 Block 数超过硬件容量，可能导致部分 Block 死锁式地永远等不到调度（尤其是涉及跨 Block 同步的场景，参考第七部分 Cooperative Launch 的约束）。

## 11.8 Kernel Fusion（算子融合）

- **Why**：多个独立 Kernel 之间的数据传递必须经过 Global Memory 落盘再读取（Kernel A 写完中间结果到显存，Kernel B 再读回来），这在 Memory Bound 场景下会浪费大量本可以避免的显存带宽（尤其当中间结果的算术强度很低、纯粹是"直传"关系时）。
- **What/How**：把多个逻辑上连续的算子合并进一个 Kernel，让中间结果直接保留在寄存器/Shared Memory 中传递给下一步计算，不落盘到 Global Memory。经典例子：FlashAttention 把 `QK^T → Softmax → ×V` 融合进一个 Kernel（第十二部分详细分析），避免了朴素实现中 Attention Score 矩阵（通常是 `O(N^2)` 规模，远大于 Q/K/V 本身）被写入再读出显存的巨大开销。
- **代价**：融合后的 Kernel 复杂度和寄存器/Shared Memory 压力都会上升，需要在"减少访存"和"资源压力升高、影响 Occupancy"之间寻找平衡点，是本部分几乎所有优化维度会同时相互牵扯的一个典型案例。

## 11.9 用 Roofline + Nsight + Micro Benchmark 形成优化闭环

- **Roofline**：确定当前 kernel 相对理论屋顶线的差距，判断优化空间和优化方向（第一部分 1.6、第九部分 9.7）。
- **Nsight Compute**：定位具体瓶颈（Stall 类型、Occupancy 限制因素、访存效率指标如 sector 利用率、Tensor Core 利用率）。
- **Nsight Systems**：从更宏观的时间线视角，观察多 Stream/多 Kernel 之间的重叠情况，定位 Host-Device 同步点、Kernel Launch 排队造成的 GPU 空闲时段。
- **Micro Benchmark（微基准测试）**：针对单一硬件行为（比如某条指令的确切延迟、某种访存模式的确切带宽），编写最小化的测试 kernel 单独测量，用来验证"我对硬件行为的理解是否正确"，也是 Citadel 等研究团队产出《Dissecting NVIDIA XXX Architecture via Microbenchmarking》系列论文的核心方法——当官方文档语焉不详时，微基准测试是获得第一手硬件行为数据最可靠的手段。

## 11.10 优化检查清单（速查）

```
□ 用 Roofline 确认 kernel 类型（Compute Bound / Memory Bound）
□ Global Memory 访问是否 Coalesced？是否用了向量化访存？
□ Shared Memory 是否存在 Bank Conflict？
□ 是否可以引入 cp.async/TMA 构建搬运-计算重叠的流水线？
□ 精度是否可以降低（FP32→TF32/BF16→FP8/FP4）而不损害正确性？
□ 是否有可以转化为 Tensor Core 矩阵乘法的计算模式？
□ Occupancy 是否是当前的限制因素？限制因素是寄存器还是 Shared Memory？
□ 是否存在不必要的 __syncthreads()，可以降级为 __syncwarp() 或彻底消除？
□ 相邻的多个 Kernel 是否可以融合，减少中间结果落盘？
□ 是否存在可以用 CUDA Graph 消除的、频繁的小 Kernel Launch Overhead？
□ 用 Nsight Compute 复测，确认 Stall 类型分布和瓶颈是否如预期发生转移
```

带着这份方法论，最后一部分我们直接分析真实的工业级/教学级 kernel 源码，从最简单的 Vector Add 一路走到 CUTLASS GEMM、FlashAttention、cuBLAS/cuDNN，把全书讲的所有硬件知识、编程模型、优化手法在真实代码中逐一对号入座。
