# 5.7 Part 6/7/8：CUDA → PTX → SASS、CUTLASS 与自建 GEMM

## 5.7.1 三层映射：把它当作实验结果

| 代码层 | PTX 可能线索 | 典型 SASS 线索 | 说明 |
|---|---|---|---|
| WMMA | `wmma.*` 或相关 `mma.sync` | `HMMA` 等 | 编译器可改变 PTX 路径 |
| inline MMA / CUTLASS MMA atom | `mma.sync.*` | `HMMA` | shape/type 依 arch 变化 |
| `ldmatrix` 路径 | `ldmatrix.*` | `LDSM` | Shared→register fragment |
| `cuda::memcpy_async` | `cp.async.*` | 常见 `LDGSTS` | 需对齐、地址空间等前提 |
| Hopper CuTe collective | TMA、`mbarrier.*`、`wgmma.*` | 常见 `HGMMA/QGMMA` | 依目标和工具链 |
| Blackwell DC CuTe collective | `tcgen05.*` | target-specific | 仅兼容 DC Blackwell |

这些不是 ABI 承诺。正确说法是：“在 CUDA X.Y、`-arch=sm_90a`、这份源码和这些编译选项下，观察到了这段 PTX/SASS。”

```bash
# 先检查编译器的虚拟 ISA 输出
nvcc -std=c++20 -arch=sm_80 -ptx gemm.cu -o gemm-sm80.ptx

# 再检查目标机器码
nvcc -std=c++20 -arch=sm_80 -cubin gemm.cu -o gemm-sm80.cubin
nvdisasm -c gemm-sm80.cubin > gemm-sm80.sass

# 记录 register/shared-memory 资源
nvcc -std=c++20 -arch=sm_80 --ptxas-options=-v -c gemm.cu
```

对 Hopper/Blackwell DC/RTX 50 分别使用实际可用的 `sm_90a`、`sm_100a`/`sm_103a`、`sm_120`。编译失败也是“这条指令不属于该目标”的有价值证据。

## 5.7.2 CUTLASS/CuTe 怎样对应硬件

```text
Device GEMM：launch、tile shape、epilogue
  → Collective mainloop：K loop、pipeline、TMA/cp.async
    → Tiled MMA：CTA/warpgroup 的计算组织
      → MMA Atom：固定 mma.sync / wgmma / tcgen05 指令抽象
        → PTX/SASS：真实 operand、指令、依赖
```

MMA Atom 并非“任意 GEMM”。它描述一个目标 ISA 支持的固定 shape、元素类型、layout 和协作约束。CuTe layout 则定义 `(thread, value) → logical matrix coordinate`：哪个线程搬哪个元素、Shared Memory 怎样 swizzle、accumulator 在 register/TMEM 的什么位置、epilogue 怎么写回。

推荐源码阅读顺序：

1. 运行最简单 TensorOp GEMM example，记录 arch、类型、tile 和性能；
2. 找到 MMA atom，写下其 PTX 家族；
3. 找 mainloop 的 copy atom/pipeline（普通 copy、`cp.async` 或 TMA）；
4. 找 epilogue，确定 accumulator 如何转换和存储；
5. 导出 PTX/SASS 验证，不要只相信模板名称。

## 5.7.3 七阶 GEMM 实战

| 版本 | 实现 | 本阶段唯一目标 |
|---|---|---|
| 1 | CUDA Core tiled GEMM | coalescing、Shared Memory 复用、CTA tile |
| 2 | WMMA GEMM | fragment、Warp 协作和数值正确性 |
| 3 | `mma.sync` + `ldmatrix` | operand tuple、Shared layout、instruction tile |
| 4 | 加 `cp.async` | 去掉寄存器中转，理解 commit/wait |
| 5 | double/triple buffer | copy/compute overlap 与 stage 生命周期 |
| 6 | Hopper TMA + WGMMA | `mbarrier` transaction、Warpgroup、异步 Tensor 完成 |
| 7 | DC Blackwell `tcgen05` | TMEM、commit/wait、目标 topology |

每次只改一个维度，并保存：高精度 reference、相同问题规模、CUDA Toolkit、编译命令、PTX/SASS、Nsight Compute 报告。否则无法判断性能变化来自 ISA、tile、精度、layout 还是测量噪声。

## 5.7.4 性能与正确性闭环

1. 先用 FP32/reference GEMM 检查绝对/相对误差；
2. warmup 后多次测量吞吐，不用单次时间下结论；
3. 检查 Tensor pipe 利用率、global/shared 吞吐、bank conflict、register、occupancy 与 stall；
4. 性能低时先判断是 memory-bound、Tensor pipe 未饱和、同步过早、寄存器压力还是错误 layout；
5. 将结果与 PTX/SASS 一起保存，保证可复现。

返回：[Tensor Core 手册入口](./00-overview.md)。同步原语请回看[第十三部分](../part13-synchronization-handbook/00-overview.md)。
