# 3.9 Hopper：Bulk Copy、TensorMap 与 TMA 使用手册

## 3.9.1 三层概念不要混在一起

| 层次 | 含义 |
|---|---|
| TMA 硬件 | Hopper 引入的异步数据搬运/地址生成能力 |
| `cp.async.bulk` | 线性 bulk copy PTX 指令族 |
| `cp.async.bulk.tensor` | 读取 TensorMap、按 1D–5D 坐标搬 tile 的 tensor copy 指令 |

日常说“TMA”时常把后两类都包括进去，但读 PTX 时必须按准确指令区分。

## 3.9.2 线性 `cp.async.bulk`

概念语法：

```ptx
// global → shared，完成记到 mbarrier transaction
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
    [dst_smem], [src_gmem], size, [mbar];

// shared → global，进入 bulk async-group
cp.async.bulk.global.shared::cta.bulk_group
    [dst_gmem], [src_smem], size;
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

基础约束通常包括：

- source/destination 16B 对齐；
- `size` 为 16B 的倍数；
- load 和 store 方向使用的 completion 机制不同；
- copy 是 weak/asynchronous memory operation，不能用普通程序顺序猜测可见性；
- 新 PTX 的 `.ignore_oob`、`.sem/.scope` 只适用于文档列出的变体，不能反推旧目标支持。

Hopper 还支持 CTA shared 与 cluster shared 之间的 bulk copy。remote destination 必须属于 cluster 内合法 peer CTA，生命周期和 cluster synchronization 另行保证。

## 3.9.3 为什么需要 TensorMap

每个 tile 都不变的部分：

```text
base address
rank、globalDim、globalStrides
boxDim、elementStrides
element type
interleave、swizzle、L2 promotion、OOB fill
```

每次 tile 改变的部分通常只有 `tensorCoords`。把前者编码成 128B descriptor，TMA 才能在硬件中做多维地址生成。

### 维度顺序

Driver API 使用**最快变化维在前**。对 C/C++ row-major `height × width`：

```cpp
uint64_t global_dim[2] = {width, height};
uint64_t global_stride[1] = {width * sizeof(T)}; // 维 1 的 byte stride
uint32_t box_dim[2] = {tile_width, tile_height};
uint32_t elem_stride[2] = {1, 1};
```

把 `{height, width}` 直接照抄进去是最常见错误之一。

## 3.9.4 Host：编码一个 2D tiled TensorMap

```cpp
#include <cuda.h>

CUtensorMap make_map(float* base,
                     uint64_t width, uint64_t height,
                     uint32_t tile_w, uint32_t tile_h) {
  CUtensorMap map{};
  constexpr uint32_t rank = 2;

  uint64_t global_dim[rank] = {width, height};
  uint64_t global_stride[rank - 1] = {width * sizeof(float)};
  uint32_t box_dim[rank] = {tile_w, tile_h};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult r = cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      rank,
      base,
      global_dim,
      global_stride,
      box_dim,
      elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("TensorMap encode failed");
  return map;
}
```

工程代码应打印 `cuGetErrorName/cuGetErrorString`，并检查：

- base 至少 16B 对齐；
- global stride 是 byte 数，且为 16B 倍数；
- box/element stride 满足 descriptor 限制；
- transfer bytes 为 16B 倍数；
- 使用 swizzle/interleave 时满足更严格约束。

`CUtensorMap` 要求 128B alignment 支持。推荐按值作为 `const __grid_constant__` kernel 参数，也可放 constant/global memory；若 descriptor 在 device 上被修改，要处理 tensormap proxy fence。

## 3.9.5 Device：2D global → shared → global

下面采用 CUDA Programming Guide 的 experimental PTX wrapper；接口名可能随 Toolkit 调整：

```cuda
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template<int H, int W>
__global__ void tma_roundtrip(
    const __grid_constant__ CUtensorMap map, int x, int y) {
  __shared__ alignas(128) float tile[H][W];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier_t bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  barrier_t::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        tile, &map, x, y, bar);
    token = cuda::device::barrier_arrive_tx(
        bar, 1, sizeof(tile));
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token)); // arrival + transaction bytes 都完成

  // 所有线程现在可消费 tile
  transform(tile, threadIdx.x);

  // generic shared writes → async proxy/TMA store
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(
        &map, x, y, tile);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
    (&bar)->~barrier_t();
  }
}
```

`wait_group_read<0>` 的重点是：TMA 已读完 shared source，buffer 才能安全覆盖；global store 对其它观察者何时可见还需上层 kernel/stream/memory-order 协议。

## 3.9.6 对应 PTX

```ptx
cp.async.bulk.tensor.2d.shared::cta.global.tile
    .mbarrier::complete_tx::bytes
    [dst_smem], [tensor_map, {x, y}], [mbar];

cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
    [tensor_map, {x, y}], [src_smem];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;
```

语法中 qualifier 的准确顺序随 PTX 版本核对；内联 PTX 不应仅凭上述概念片段复制。

## 3.9.7 mbarrier transaction：为什么同时有 arrival 与 bytes

```text
线程条件：所有参与者 arrive
事务条件：TMA complete_tx 的累计 bytes 达到 expect_tx
                    ↓
               barrier phase 完成
```

只满足其中一个都不能消费 tile。常见 protocol：

1. 一个线程初始化 barrier；
2. async-proxy fence + CTA sync 让初始化可见；
3. issuer 发 TMA；
4. issuer `arrive_tx(expected_bytes)`，其它线程普通 arrive；
5. 每个线程用本 phase token wait；
6. 消费后进入下一 phase。

expected bytes 写小会过早完成，写大会永久等待。多 stage 时每个 stage 通常持有独立 barrier/state，避免 phase 混乱。

## 3.9.8 OOB 与尾块

global→shared tiled TMA 允许 tile 部分越界，并按 TensorMap fill mode 填 0 或 OOB-NaN；tile 起点可为负。shared→global 时越界部分可被丢弃，但左上/起始坐标的负值受更严格限制。

自动 fill 的价值：

- GEMM 的 M/N/K 尾块不再逐 lane 分支；
- 卷积 halo 可用 descriptor/mode 表达；
- 但算法要确认 0 是否为正确中性元。

## 3.9.9 TMA swizzle

配置 swizzle 后，TMA 按 swizzled layout 写 shared。消费者必须用匹配布局：

```text
global row-major tile
  → TMA address generation
  → shared swizzled tile
  → WGMMA/ldmatrix 按同一 layout 读取
```

不要先让 TMA swizzle，随后用普通 `tile[row][col]` 假设数据仍线性。CUTLASS/CuTe 的 layout atom 正是在编译期保持 producer/consumer 映射一致。

## 3.9.10 Cluster multicast

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.multicast::cluster
    [dst], [map, {x, y}], [remote_mbar], ctaMask;
```

一份 global tile 可写入 mask 指定的多个 CTA shared memory。收益来自减少重复 L2/HBM 请求；代价是：

- kernel 必须以 cluster 方式 launch；
- destination/barrier 地址和 CTA rank 映射正确；
- 每个目标 CTA 的 shared 生命周期受保护；
- TMA transaction 完成不自动等价于所有 CTA cluster-sync。

## 3.9.11 L2 prefetch

```ptx
cp.async.bulk.prefetch.L2.global [src], size;
cp.async.bulk.prefetch.tensor.2d.L2.global.tile
    [tensor_map, {x, y}];
```

prefetch 是弱提示，没有供程序等待的“数据必在 L2”保证。应在访问规律明确、距离可调且不会污染 cache 时实测。

## 3.9.12 什么时候用 CuTe/CUTLASS

生产 GEMM/attention 通常优先 CuTe/CUTLASS，因为它们把以下关系放进类型/layout：

- TensorMap shape/stride；
- shared swizzle；
- TMA copy atom；
- mbarrier pipeline state；
- WGMMA operand layout；
- cluster multicast。

建议先完成本章的最小 roundtrip，理解 completion 后再读框架封装；否则模板错误很难归因。

mbarrier phase、transaction bytes、async proxy 与 cluster 交接的规范化解释见[第十三部分 Async Pipeline Synchronization](../part13-synchronization-handbook/03-async-pipelines.md)；WGMMA 与 TMA 的组合见[第五部分 Hopper WGMMA](../part05-tensor-core-handbook/05-hopper-wgmma.md)。

下一篇：[Blackwell TMA 功能矩阵](./06-blackwell-tma.md)。
