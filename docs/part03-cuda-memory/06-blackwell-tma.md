# 3.9B Blackwell TMA：Hopper TMA 的功能扩展

> 这里的“Blackwell TMA”指公开 PTX/CUDA 文档中在 Hopper tensor-copy 基础上、对 Blackwell target 新增或重点扩展的能力。不要把芯片内部模块表中的简称直接当作可编程 PTX 名称。

## 3.9B.1 先看总图

| 类别 | 公开 PTX 入口 | 搬运/操作方向 | 完成机制 |
|---|---|---|---|
| 线性 bulk copy | `cp.async.bulk` | global↔shared、CTA↔cluster 的合法变体 | load/cluster 常用 mbarrier；store 常用 bulk group |
| 线性 bulk reduce | `cp.reduce.async.bulk` | shared→global 或 shared→cluster | bulk group 或 mbarrier |
| 多维 tensor copy | `cp.async.bulk.tensor.{1d..5d}` | global→shared/cluster、shared→global | mbarrier 或 bulk group |
| tensor reduce | `cp.reduce.async.bulk.tensor.{1d..5d}` | shared→global | bulk group |
| prefetch | `cp.async.bulk.prefetch(.tensor)` | global→L2 hint | 无可等待完成保证 |
| cluster multicast | tensor copy `.multicast::cluster` | global→多个 CTA shared | mbarrier transaction |
| CTA pair | tensor copy `.cta_group::2` | 与 paired CTA/peer mbarrier 协作 | mbarrier |

图片中的 “DLS” 并不是当前公开 PTX ISA 的独立 mnemonic。若它指 distributed/cluster load-store，程序员可见的接口是 `.shared::cluster`、multicast、remote mbarrier、CTA group 等；不要凭内部分类虚构 `tma.dls` 指令。

## 3.9B.2 Bulk copy：Blackwell 仍保留的基础

```ptx
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
    [dst], [src], size, [mbar];

cp.async.bulk.global.shared::cta.bulk_group
    [dst], [src], size;
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

PTX 9.2 增加 `.ignore_oob` 线性 load 变体：首尾各最多忽略 0–15 个越界字节，目标对应字节值不确定；地址仍需 16B 对齐、size 仍为 16B 倍数。它不是 tensor zero-fill。

PTX 9.3 又给部分 bulk 指令增加 `.weak` 或 `.relaxed`、`.scope` 与 `.type`。这是内存模型扩展，不是“更快模式”；只有在确实需要跨观察者 ordering 时按语义选择。

## 3.9B.3 Bulk reduce

`cp.reduce.async.bulk` 将 source 数组逐元素归约进 destination，而不是先 copy 再启动另一个 reduction kernel：

```ptx
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32
    [dst_global], [src_shared], size;
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

### shared→global 的合法 op/type 组合

| `redOp` | 类型 |
|---|---|
| `add` | `u32/s32/u64/f32/f64/f16/bf16` |
| `min/max` | `u32/s32/u64/s64/f16/bf16` |
| `inc/dec` | `u32` |
| `and/or/xor` | `b32/b64` |

`f16/bf16 add` 使用 `.noftz` 形式并保留 subnormal；当前 `f32 add` 会把 subnormal 输入/结果 flush 为带符号零。每个 destination element 的 reduction 是原子的，但整个数组不是一个不可分割事务。

### shared→cluster shared

该方向使用 mbarrier completion，类型/operation 子集更小：

- add：`u32/s32/u64`；
- min/max：`u32/s32`；
- inc/dec：`u32`；
- bitwise：`b32`。

适合 cluster CTA 将局部片段直接归并到 remote shared buffer，但需要单独保证目标 CTA 生命周期和后续消费者同步。

## 3.9B.4 Tensor element type

公开 TensorMap/PTX 类型包括：

- bit：`b32/b64`；
- integer：`u8/u16/u32/s32/u64/s64`；
- floating/alternate：`f16/bf16/tf32/f32/f64`；
- Blackwell sub-byte：`b4x16`、`b4x16_p64`、`b6x16_p32`、`b6p2x16`。

Driver API 对应 U4/U6 枚举：

```text
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
```

这些名字描述的是 16 个 sub-byte element 的 packing/alignment，不是 `float4`：

- 16U4 ALIGN8B：16×4bit 紧密占 8B；
- 16U4 ALIGN16B：8B 有效数据扩展到 16B slot；
- 16U6 ALIGN16B：12B 有效数据扩展到 16B slot；
- store 方向的 `b6p2x16` 可丢弃每元素高 2bit padding，打包为 U6。

关键限制包括：

- 某些 padded U4/U6 要求 global base/stride 32B 对齐；
- dim0 大小、boxDim[0]、首坐标有专门倍数约束；
- OOB-NaN 不支持 sub-byte 类型；
- 并非所有 load/store 方向都支持所有 sub-byte 类型；
- `sm_120a` 的 cluster→global sub-byte 支持受限；
- 必须按目标 PTX 的 restriction 表逐项核对。

## 3.9B.5 1D–5D tile mode

```ptx
cp.async.bulk.tensor.1d.shared::cta.global.tile...
cp.async.bulk.tensor.2d.shared::cta.global.tile...
...
cp.async.bulk.tensor.5d.shared::cta.global.tile...
```

`.tile` 保留源 tensor 的多维次序，将所选 bounding box 线性/按 swizzle 落入 shared。它适合 GEMM tile、batched tensor、3D/5D 激活等规则区域。

使用原则：

1. Host 以 `cuTensorMapEncodeTiled` 编码；
2. kernel 传入 1–5 个 signed 32-bit 坐标；
3. load 用 mbarrier transaction；
4. store 用 bulk group；
5. fill、swizzle、interleave 来自 descriptor。

## 3.9B.6 `tile::gather4` 与 `tile::scatter4`

两者只适用于 2D tensor，访问四个不连续 row：

```text
tensorCoords = {col, row0, row1, row2, row3}
```

- `gather4`：global 的四行组合成一个 shared 2D tile，常用于 load；
- `scatter4`：一个 shared tile 分散写到 global 的四行，常用于 store；
- 不支持 interleave layout；
- 其它 tile-mode 对齐与 bounding-box 规则仍适用。

示意：

```ptx
cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global
    .mbarrier::complete_tx::bytes
    [dst], [map, {col, r0, r1, r2, r3}], [mbar];

cp.async.bulk.tensor.2d.tile::scatter4.global.shared::cta
    .bulk_group
    [map, {col, r0, r1, r2, r3}], [src];
```

`gather4`/`scatter4` 在 PTX 8.6 加入，但 target notes 对 shared::cta、shared::cluster 和不同 Blackwell family 的要求不同；存在 qualifier 不代表 Hopper 可执行。

## 3.9B.7 `im2col`

普通 `im2col` 让 TMA 把 N/D/H/W 中的多个像素与 channel 数据展开成 shared 中的一维 column，省去线程循环和边界分支。

Host 使用：

```cpp
cuTensorMapEncodeIm2col(
    &map, type, rank, base,
    global_dim, global_stride,
    pixel_box_lower, pixel_box_upper,
    channels_per_pixel, pixels_per_column,
    element_stride,
    interleave, swizzle, l2_promotion, oob_fill);
```

Device 还传入 offset：

```ptx
cp.async.bulk.tensor.5d.shared::cta.global.im2col
    .mbarrier::complete_tx::bytes
    [dst], [map, {n,c,d,h,w}], [mbar],
    {off_w, off_h, off_d};
```

适合卷积 lowering，但“硬件 im2col”不保证整个 convolution 最优；direct convolution、implicit GEMM 的 tile 复用和 framework layout 仍需实测。

## 3.9B.8 `im2col::w` 与 `im2col::w::128`

Blackwell wide 模式只沿 W 访问，保持 H/D 不变。Host 使用 `cuTensorMapEncodeIm2colWide`：

```cpp
cuTensorMapEncodeIm2colWide(
    &map, type, rank, base,
    global_dim, global_stride,
    lower_w, upper_w,
    channels_per_pixel, pixels_per_column,
    element_stride, interleave,
    CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128, // 或 W
    swizzle, l2_promotion, oob_fill);
```

区别：

- `im2col::w`：加载元素数由 `pixelsPerColumn` 决定；
- `im2col::w::128`：固定 128 个元素，descriptor 的 `pixelsPerColumn` 被忽略；
- `wHalo`/`wOffset` 是 device 指令参数；
- W128 每 32 个元素插入 `wHalo` 个 halo 元素；
- 必须启用合法 swizzle；none 和某些 128B+flip 组合非法；
- 不支持 interleave layout。

## 3.9B.9 新 swizzle atomicity

传统 32/64/128B swizzle 以 16B chunk 重排。Blackwell 扩展可在 128B span 内保持更大原子块：

| 模式 | 含义 |
|---|---|
| `128B_ATOM_32B` | 重排时每个 32B 保持完整 |
| `128B_ATOM_32B_FLIP_8B` | 32B atomic，并在隔行交换 16B 内相邻 8B |
| `128B_ATOM_64B` | 重排时每个 64B 保持完整 |

用途是同时匹配低精度 packed 数据、Tensor Core operand 与 shared bank。`FLIP_8B` 只允许特定 store 方向，wide im2col 也禁用部分组合。选择依据必须是消费者 layout，不是“数值越大越好”。

## 3.9B.10 CTA pair / `.cta_group::2`

```ptx
cp.async.bulk.tensor.1d.shared::cta.global.tile
    .mbarrier::complete_tx::bytes.cta_group::2
    [dst], [map, {x}], [peer_mbar];
```

`.cta_group::2` 允许 CTA pair 中的 TMA 操作按双 CTA 协作规则通知本 CTA或 peer CTA 的 mbarrier。它服务于 Blackwell CTA-pair Tensor pipeline：

- kernel launch/topology 必须形成合法 pair；
- shared destination 与 mbarrier 所属 CTA 必须满足指令规则；
- peer barrier 不是普通指针共享；
- `.cta_group::2` 只改变 TMA 的协作/通知范围，不替代 cluster barrier；
- 常与 `tcgen05` CTA-group 计算配套，但两者的 completion 仍是不同协议。

## 3.9B.11 Tensor reduce

```ptx
cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
    [map, {x, y}], [src];
```

它按 TensorMap 地址将 shared tile 逐元素归约到 global tensor。operation/type 组合与 bulk reduce 类似，但需查 tensor-reduce 的独立 restriction；sub-byte tensor 类型不支持 reduce。

## 3.9B.12 功能如何选

| 需求 | 模式 |
|---|---|
| 连续大块 | `cp.async.bulk` |
| 规则多维 tile + 尾块填充 | tiled TMA |
| 四个不连续 row | gather4/scatter4 |
| 卷积 NDHW 展开 | im2col |
| 只沿 W 的宽窗口/halo | im2col::w/W128 |
| 多 CTA 复用同一 tile | multicast |
| CTA pair producer/consumer | `.cta_group::2` |
| 搬运同时归约 | bulk/tensor reduce |
| 低精度 packed operand | U4/U6 TensorMap + 合法 swizzle |

下一篇：[cp.async/TMA 流水线实战](./07-pipeline-examples.md)。
