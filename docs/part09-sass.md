# 第九部分 SASS（高级）

> 这一部分真正开始读机器码。这是绝大多数 CUDA 教程止步的地方，但也是从"会写 CUDA"到"能读懂/调优工业级 kernel"的分水岭——Nsight Compute 的很多深层分析结果，最终都要落到"看某一行 SASS 为什么会 stall"这个层面才能真正定位问题。

## 9.1 SASS 是什么

SASS（Streaming ASSembler）是 NVIDIA GPU 真正执行的机器码的汇编表示，与架构强绑定（不同 `sm_XX` 的 SASS 指令集、编码格式都不同，也没有正式公开的完整 ISA 手册，社区主要靠 `nvdisasm`/`cuobjdump` 反汇编结果和逆向研究积累认知）。查看方式：

```bash
cuobjdump -sass a.out          # 从可执行文件/cubin 中反汇编出 SASS
nvdisasm -c kernel.cubin       # 另一个反汇编工具，输出风格略有不同
```

### 9.1.1 最小实操：CUDA `__fmaf_rn` → PTX `fma` → SASS `FFMA`

配套 case：[`src/part09-sass/case01_ffma.cu`](../src/part09-sass/case01_ffma.cu)。目标是在 Blackwell（本仓库实测为 Thor / `sm_110`）上走通一遍“源码 → PTX → SASS”，不强求性能。

CUDA（内核核心一行）：

```cuda
d[i] = __fmaf_rn(a[i], b[i], c[i]);  // fused multiply-add，round-to-nearest
```

构建与查看产物：

```bash
cmake -S . -B build -DNVGPU_CUDA_ARCH=native
cmake --build build -j --target part09_case01_ffma part09_sass_inspect
ctest --test-dir build -R part09_ --output-on-failure

# PTX / SASS：
#   build/src/part09-sass/artifacts/part09_case01_ffma.ptx
#   build/src/part09-sass/artifacts/part09_case01_ffma.sass
```

本机（CUDA 13 / Thor）一次实测片段：

**PTX**（`cicc` 输出的虚拟 ISA）：

```ptx
ld.global.nc.f32  %f1, [%rd10];
ld.global.nc.f32  %f2, [%rd11];
ld.global.nc.f32  %f3, [%rd12];
fma.rn.f32        %f4, %f1, %f2, %f3;
st.global.f32     [%rd13], %f4;
```

**SASS**（`ptxas` 针对本机 arch 生成、经 `cuobjdump --dump-sass`）：

```text
LDG.E.CONSTANT R2, desc[UR4][R2.64] ;
LDG.E.CONSTANT R5, desc[UR4][R4.64] ;
LDG.E.CONSTANT R7, desc[UR4][R6.64] ;
FFMA R11, R2, R5, R7 ;
STG.E desc[UR4][R8.64], R11 ;
```

对照阅读：

| 层 | 看到什么 |
|---|---|
| CUDA | `__fmaf_rn(a,b,c)` 明确要求融合乘加 |
| PTX | `fma.rn.f32`：虚拟指令 + 舍入模式；访存仍是 `ld.global` / `st.global` |
| SASS | `FFMA`：真实机器助记符；Blackwell 上 global 常表现为带 `desc[...]` 的 `LDG`/`STG` |

注意：SASS 助记符、地址模式（如 `desc[UR4][...]`、`.CONSTANT`）随架构与 Toolkit 变化；换 `-arch` 或换卡后以你本机 `*.sass` 为准，不要把上面片段当成所有 Blackwell 的固定模板。运行时 case 只检查 `1.5*2+3=6`；读码请看 `*_inspect` 产物。

## 9.2 Instruction Format：一条 SASS 指令长什么样

一条典型的 SASS 指令输出（以近似 Ampere/Hopper 风格示意）大致包含：地址、控制码、助记符、操作数：

```
/*0058*/  @!P0 LDG.E.128.SYS R4, [R2+0x10] ;      /* 0x...  控制码(隐藏在编码中) */
/*0060*/       IADD3 R6, R6, 0x4, RZ ;
/*0068*/       FFMA R8, R4, R5, R8 ;
```

- `@!P0`：谓词（Predicate）执行前缀，只有当 P0 寄存器为假时才真正执行这条指令——这是 GPU 处理分支的常见手法之一：把简单的 if 语句编译成谓词化指令而不是真正的跳转，避免分支带来的额外控制流开销（详见 9.2.1）。
- `LDG.E.128.SYS`：全局内存加载，`.E`表示扩展地址模式，`.128` 表示一次搬运 128 bit（16 字节，即 `float4` 向量化访存的直接体现），`.SYS` 表示内存一致性范围（system scope）。
- 每条指令背后还隐藏着一段**控制码（Control Code）**，不会以独立助记符的形式出现在反汇编的主体文本里，而是编码在指令二进制中的额外字段，或者（如 Maxwell/Pascal 时代）以独立的"控制指令"形式，每 3 条真实指令前插入一条，供 Warp Scheduler 使用。

### 9.2.1 谓词化（Predication）：另一种处理分支的手段

对于 Warp 内分支体很短的情况（比如一个简单的三目表达式），编译器常常不会生成真正的跳转指令，而是把两条路径的指令都编译出来，各自加上互斥的谓词前缀（`@P0` / `@!P0`），让硬件"执行两条路径的指令，但被谓词屏蔽的那条不产生实际效果（不写寄存器/不访存）"。这样虽然浪费了一些指令槽位（两条路径的指令都占用了发射带宽），但避免了分支预测失败/跳转本身的开销，对于短分支体是更划算的选择；只有当分支体足够长时，编译器才会退化为使用真正的跳转（`BRA`）+ 第一部分讲的 SIMT 栈/Reconvergence 机制。

## 9.3 控制码：Stall / Yield / Barrier / Reuse

延续第四部分的铺垫，这里给出更完整的控制码字段构成（现代架构上，通常打包为一个约 23 位的控制字，逻辑上对应）：

| 字段 | 位宽 | 作用 |
|---|---|---|
| Stall Count | 4 bit | 固定延迟指令：发射后需要等待的周期数（0-15），覆盖大多数 ALU 流水线延迟 |
| Yield Flag | 1 bit | 提示 Warp Scheduler："这条指令之后适合切换到别的 Warp"，常见于伴随 Scoreboard 等待的指令 |
| Write Barrier Index | 3 bit | 这条指令（变延迟操作，如 `LDG`）完成时，点亮哪个编号（0-5，即 6 个硬件 Scoreboard）的 Barrier |
| Read/Wait Barrier Mask | 6 bit ×2 | 这条指令在发射前必须等待哪些 Barrier 已点亮；执行完成后释放哪些 Barrier |
| Reuse Flag | 6 bit（每源操作数） | 提示"操作数复用缓存（Operand Reuse Cache）"可以直接复用上一条指令已经取出的寄存器值，避免重复读取寄存器堆的一个端口，缓解寄存器 Bank 冲突 |

### 9.3.1 为什么会有 Reuse Flag

寄存器堆的读端口数量是有限的（不可能给每个操作数一个独立端口，否则面积/功耗太高），如果连续几条指令都要读同一个寄存器（比如循环展开后的多条 `FFMA` 共用同一个乘数），每次都真正访问寄存器堆会造成端口争用（寄存器 Bank Conflict，与 Shared Memory Bank Conflict 是完全不同层面的概念，但思路类似）。**Reuse Flag** 允许 `ptxas` 在生成指令时告诉硬件："这个操作数值和上一条指令用的某个操作数相同，直接从一个小的操作数缓存里拿，不用真正访问寄存器堆"，从而在不增加寄存器堆物理端口的前提下，缓解连续指令复用同一寄存器造成的瓶颈。研究表明，去掉编译器精心设置的 Reuse Flag 会导致某些访存密集型 kernel 出现明显的性能下降，这也是为什么手写/优化 SASS（如历史上的 maxas 项目对 Maxwell SGEMM 的手工调度）非常关注这个标志位。

## 9.4 Dependency 与 Scoreboard（承接第四部分，补充细节）

第四部分讲了 Scoreboard 的基本机制，这里补充两个实践中重要的细节：

1. **6 个 Barrier 是稀缺资源**：如果一段代码中同时有超过 6 个"变延迟"操作在飞行中（比如展开了很多路的异步加载），编译器不得不让某些操作复用同一个 Barrier 编号，这会导致等待其中一个操作完成的指令，被迫也等待共享同一 Barrier 的其它操作——过度的循环展开、过多同时在飞行的异步操作，反而可能因为 Barrier 资源耗尽引入不必要的等待，这是"展开/流水线级数并非越多越好"的一个具体硬件原因。
2. **`DEPBAR` 显式依赖屏障指令**：当编译器需要显式地让一条指令等待某个 Barrier，但又不方便把等待信息编码进目标指令自身的控制码时（或是为了修复某些同步引入的 WAR 写后读冒险），会插入独立的 `DEPBAR` 指令。

## 9.5 Register Allocation：为什么同样的 CUDA 代码会生成不同的 SASS

寄存器分配是 `ptxas` 内部影响最大的优化决策之一，以下几个因素都会导致同一份 CUDA C++ 源码，在不同条件下生成截然不同的 SASS：

- **`--maxrregcount` / `__launch_bounds__`**：显式限制每线程最大寄存器数，会迫使编译器在"寄存器够用、性能好"和"寄存器紧张、需要溢出到 Local Memory 但换来更高 Occupancy"之间做取舍，直接影响生成代码的形态（第十一部分会展开这个权衡的实战判断方法）。
- **不同架构目标（`-arch=sm_80` vs `sm_90a`）**：不同架构上物理执行单元、可用指令集（如是否支持 `wgmma`）、Scoreboard 数量等硬件参数不同，`ptxas` 会生成完全不同的指令选择和调度方案。
- **不同 CUDA Toolkit/驱动版本**：`ptxas` 本身作为一个持续迭代的编译器，其内部优化 pass（指令调度算法、寄存器分配启发式）会随版本更新而改变，同一份源码用不同版本工具链编译，SASS 也可能不同——这是做严格性能回归测试时需要固定编译器版本的原因之一。
- **编译期是否已知的常量/循环边界**：如果 kernel 里的循环次数、数组大小等是编译期常量，编译器有更大空间做循环展开、寄存器分配优化；一旦这些变成运行时变量，很多优化机会会消失。
- **内联决策**：函数是否被内联直接影响寄存器分配的作用范围（内联后调用者和被调用者共享寄存器分配决策空间），`__forceinline__`/`__noinline__` 可以显式干预。

## 9.6 Instruction Scheduling：`ptxas` 如何排布指令顺序

`ptxas` 会在满足数据依赖（不改变程序语义）的前提下，重排指令顺序，尽量做到：

1. 把一条变延迟指令（如 `LDG`）和它的消费者指令之间，插入足够多的无关指令，让延迟被"自然填充"而不需要真正的 stall（这是"指令级并行 ILP"在编译期被主动构造出来的过程）。
2. 均衡不同执行单元（ALU、LD/ST、SFU、Tensor Core）的指令分布，充分利用 Dual Issue（第四部分 4.5 节）的机会。
3. 合理复用 Reuse Flag，减少寄存器 Bank 争用。

这也是为什么"手工重排 SASS 指令顺序"（如 CuAsmRL 这类用强化学习自动搜索更优 SASS 调度的研究工作）在某些场景下还能比 `ptxas` 的默认调度获得额外的性能提升——`ptxas` 的调度算法是通用启发式，不一定对每个具体 kernel 都是全局最优。

## 9.7 用 Nsight Compute 看 SASS、分析 Stall

Nsight Compute 是分析 SASS 级别性能问题的标准工具，几个关键的分析入口：

- **Source/SASS 视图**：可以逐指令查看每条 SASS 指令消耗的周期数、被采样到"warp 处于该指令时正在 stall"的比例，直接定位热点指令。
- **Warp State Statistics**：把 Stall 归因到具体原因（常见的几类）：
  - `stall_long_scoreboard`：等待一次长延迟操作（典型是 Global Memory 访问）的 Scoreboard——如果这个占比很高，说明访存延迟没有被充分隐藏，需要检查 Occupancy 是否足够、或访存模式是否高效（回到第三部分的 Coalescing 诊断）。
  - `stall_short_scoreboard`：等待一次相对短延迟的操作（如 Shared Memory 访问、`cp.async` 相关等待）。
  - `stall_barrier`：等待 `bar.sync`（`__syncthreads()`），说明 Block 内不同 Warp 的工作量不均衡，快的 Warp 在等慢的 Warp。
  - `stall_not_selected`：这个 Warp 已经就绪，但 Warp Scheduler 那个周期选择了别的 Warp——通常说明 Occupancy 过高、Warp 数量已经超出必要，可以考虑适当降低。
  - `stall_wait`：等待固定延迟指令的 Stall Count 走完。
  - `stall_math_pipe_throttle` / `stall_drain` 等：特定执行单元吞吐已经打满，或者流水线排空阶段的等待。
- **Occupancy 分析**：结合寄存器/Shared Memory 用量，给出理论 Occupancy 上限与实际达到值的对比，定位是哪种资源在限制并发度。
- **Roofline Chart**：Nsight Compute 内置的 Roofline 图，直接把第一部分讲的理论模型和实测数据点画在一起，一眼看出当前 kernel 落在 Compute Bound 还是 Memory Bound 区域，以及距离屋顶线还有多少空间。

## 9.8 小结：SASS 分析的核心方法论

读 SASS/用 Nsight Compute 分析性能，本质上是把前面第一到第八部分讲的所有硬件知识，串联起来做"侦探式"的证据链推理：

```
观察到的现象（如 stall_long_scoreboard 占比高）
      ↓ 对照第四部分：Scoreboard 机制
定位到具体是哪条/哪类指令在等（如 LDG）
      ↓ 对照第三部分：这次访存是否 Coalesced？是否该用 cp.async/TMA 重叠？
定位到根因（如访存模式差 / Occupancy 不足以掩盖延迟）
      ↓ 对照第二部分：调整 Block 大小 / 寄存器用量以提升 Occupancy
      ↓ 对照第三部分：改善访存模式 / 引入异步流水线
验证：重新编译、重新用 Nsight Compute 测量，确认 stall 占比下降、总耗时降低
```

这套"现象 → 硬件机制 → 根因 → 具体优化手段 → 验证"的闭环，正是第十一部分要系统化整理的性能优化实战方法论。而在此之前，我们先在第十部分把前面所有的硬件演进线索，按时间顺序完整串一遍——因为只有理解了每一代硬件"新增了什么、为什么增加、软件怎么支持、性能提升在哪"，第十一部分的优化手段才有明确的"针对哪一代硬件、利用哪一个新特性"的具体抓手。
