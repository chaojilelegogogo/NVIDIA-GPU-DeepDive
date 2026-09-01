# 3.2–3.5 访问模式：Coalescing、Transaction、Bank 与 Swizzle

## 3.2.1 统一视角：逻辑访问如何膨胀成物理事务

程序写的是“每线程读一个元素”，硬件处理的是若干固定粒度的 sector/cache transaction。性能问题通常可写成：

```text
requested bytes（算法真正需要）
        ↓ 地址合并
transferred bytes（存储层真正搬运）
        ↓ cache / bank / replay
实际占用的带宽和周期
```

`requested / transferred` 越低，越多带宽被浪费。不要把某个架构上的“128B transaction”当成永恒常数；不同 cache 层、指令与架构的分段方式不同，但“覆盖尽量少的对齐段”这个原则不变。

## 3.2.2 Global Memory Coalescing

理想模式：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[i];
```

同一 warp 的活跃 lane 访问连续 `float`，地址范围紧凑。反例是 stride 大或散乱索引：

```cuda
float x = input[i * stride];
```

即使每个线程只取 4B，32 个地址若落在 32 个不同 sector，就会产生大量低利用率事务。

### AoS 与 SoA

```cpp
struct Particle { float x, y, z, mass; };
```

若 warp 只读取 `x`，AoS 访问步长为 16B；SoA 的 `x[]` 则连续。选择布局要看最常见的**同一条指令跨 lane 的地址集合**，而不是只看单个对象是否紧凑。

## 3.3 Alignment

跨越边界的连续区域可能被拆成更多事务。检查三层对齐：

1. allocation base：`cudaMalloc` 通常提供足够的基础对齐；
2. subview offset：切片、header、指针偏移可能破坏对齐；
3. instruction width：`float4`、`cp.async`、bulk/TMA 各有自己的地址和大小约束。

向量化类型不能“创造”对齐。把未对齐地址强转为 `float4*` 可能是未定义行为或退化访问。

## 3.4.1 Shared Memory Bank Conflict

Shared Memory 常按 32 个 bank 组织，连续 32-bit word 轮转映射到 bank。近似模型：

```text
bank = (byte_address / bank_width) mod 32
```

同一 warp 指令中：

- 不同 lane 访问不同 bank：并行；
- 多 lane 读取同一地址：可广播；
- 多 lane 访问同一 bank 的不同地址：产生 conflict/replay。

经典转置：

```cuda
__shared__ float tile[32][32]; // 按列访问会形成 32-way conflict
__shared__ float tile_pad[32][33]; // padding 打散行首 bank
```

Padding 简单有效，但 Tensor Core operand 的布局更复杂，常用 swizzle。

## 3.4.2 Swizzle 是什么

Swizzle 是从逻辑坐标到 shared 地址的可逆重排：

```text
logical (row, col)
  → xor/permutation
  → physical shared address
```

目标是让计算阶段的一条 `ldmatrix`、WGMMA 或普通 load/store 覆盖不同 bank，同时保留可计算的布局。软件手写 swizzle 时，生产者和消费者必须使用同一映射；TMA swizzle 则把 global→shared 落位重排交给 TensorMap/TMA 硬件。

## 3.4.3 TMA swizzle

常见 TensorMap swizzle 模式包括：

- none；
- 32B、64B、128B span，以 16B chunk 为基本 atomicity；
- Blackwell 相关扩展：128B span 下保持 32B atomic、32B atomic + 8B flip、64B atomic；
- PTX 新版本还定义 96B swizzle，但有严格类型、box 和 interleave 限制。

“128B swizzle”不表示每次一定搬 128B，而是描述 shared address permutation 的周期/span。配置步骤：

1. 先确定消费者每条指令如何读 shared；
2. 选择使该访问无 conflict 的 swizzle；
3. 确保 innermost box bytes 不超过所选 span 等约束；
4. shared base 按规范对齐，通常 TMA 例程使用 128B；
5. 用同一 swizzle 的布局规则解释 shared 数据，而不是按原始 row-major 下标直接读取。

示意：

```cpp
cuTensorMapEncodeTiled(
    &map, type, rank, global,
    global_dim, global_stride,
    box_dim, element_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

## 3.4.4 Interleave 与 swizzle 不同

- **Interleave** 描述 global tensor 本身的通道/元素组织，例如 16B/32B interleaved layout；
- **Swizzle** 主要描述 TMA 在 shared memory 中如何重排以避开 bank conflict。

二者存在合法组合约束。例如 32B interleave 要求更严格的 global address/stride 对齐，并要求匹配的 32B swizzle。不能把它们都笼统叫“layout”后任意组合。

## 3.4.5 OOB fill

TensorMap 的 bounding box 可跨出 global tensor 边界。TMA load 可自动填：

- zero fill：越界元素写 0，适合 GEMM/卷积尾块；
- OOB-NaN fill：写特殊 NaN，主要用于浮点越界诊断/传播语义。

这消除了每个线程的边界分支，但不会放宽 descriptor 的 alignment、stride、box 等约束。Blackwell sub-byte 类型不支持 OOB-NaN。

## 3.5 Transaction 利用率与诊断方法

1. 用 Nsight Compute 看 global sectors/request 与 requested/actual throughput；
2. 看 shared bank conflict/replay 指标；
3. 比较 swizzle 前后同一消费者的 shared transaction；
4. 检查向量宽度是否真的生成预期 load/store；
5. 对尾块单独测试，确认 OOB fill 与算法中性元一致。

下一篇：[数据搬运机制的完整演进](./03-data-movement-evolution.md)。
