# 5.2 Part 1：为什么 GPU 需要 Tensor Core

## 5.2.1 标量 FMA 的瓶颈

普通 CUDA Core 做 GEMM 的内层类似：

```cuda
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
}
```

这段代码每次循环只表达一个标量 FMA。神经网络训练/推理包含海量、规则、可分块的 GEMM/卷积；若继续用标量 FMA，指令取指、调度和寄存器读取的控制开销会重复许多次。Tensor Core 将“固定小矩阵的许多 FMA”封装为一条矩阵指令，并用专用阵列并行完成。

## 5.2.2 Roofline：为什么只有算得快还不够

Tensor Core 提升峰值算力后，数据供给常成为瓶颈：

```text
算术强度 = 有效 FLOPs / 从慢存储搬运的字节数

低算术强度：受 HBM/L2 带宽限制，Tensor Core 可能空闲
高算术强度：受 Tensor Core 吞吐限制，tile/pipeline 变得关键
```

所以 Tensor Core ISA 的演进不只有 MMA 指令：

- Ampere `cp.async`：global→shared 不再经过寄存器；
- Hopper TMA：由专用 DMA 按描述符搬整个 tile；
- Hopper WGMMA：从 Shared Memory descriptor 消费 operand；
- Blackwell TMEM：结果不必长期占用通用寄存器。

## 5.2.3 Mixed Precision：用位宽交换吞吐和带宽

| 格式/时期 | 工程动机 | 需要注意 |
|---|---|---|
| FP16 | 输入更小、吞吐更高 | 动态范围有限，训练常需 loss scaling |
| INT8/INT4 | 推理可接受更低精度 | 量化/反量化和累加精度 |
| TF32 | 保留 FP32 指数范围，降低尾数精度 | 不是内存格式替换；是 Tensor Core 计算路径 |
| BF16 | 接近 FP32 的指数范围 | 尾数精度更低 |
| FP8/FP4 | 更高吞吐和更少带宽 | scale、格式、误差与硬件支持必须明确 |

低精度并非自动正确。你必须定义输入量化、accumulator 类型、epilogue、误差指标和测试集。尤其 FP4/FP8 常需要 block/micro-block scale；scale 也是数据布局和 MMA operand contract 的一部分。

## 5.2.4 编程模型如何随硬件演进

```text
Volta：操作数和 accumulator 都在寄存器
  问题：寄存器和 Warp tile 限制了规模
Ampere：shared tile 通过 ldmatrix 高效分发到寄存器
  问题：数据搬运和地址计算仍消耗 Warp
Hopper：TMA 搬 tile，WGMMA 从 shared descriptor 异步计算
  问题：大 accumulator 重新压垮 register file
Blackwell DC：tcgen05 将 accumulator 放入 TMEM
```

这条链比死记 API 更重要：每项新特性都在消除一个已被前代算力放大的瓶颈。

下一篇：[Volta WMMA 与第一代 MMA](./03-volta-wmma.md)。
