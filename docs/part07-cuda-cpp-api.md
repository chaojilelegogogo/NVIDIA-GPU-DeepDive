# 第七部分 CUDA C++ API

> 这一部分讲 API，但核心目标不是罗列函数签名，而是讲清楚 **CUDA C++ API → PTX → SASS** 这三层关系：每一个高层 API 调用，最终会变成什么样的 PTX 指令序列，又会被 `ptxas` 编译成什么样的 SASS。理解这条链路，你才能在"用什么 API"和"底层硬件行为是什么"之间自由切换，而不是把 API 当黑盒。

## 7.1 三层关系总览

```
┌─────────────────────────┐
│   CUDA C++ / Runtime API │   你写的代码：cudaMalloc, kernel<<<>>>, cuda::pipeline, wmma::...
└───────────┬──────────────┘
            │ nvcc 前端编译（cicc）
            ▼
┌─────────────────────────┐
│         PTX (虚拟ISA)      │   中间表示：ld.global, mma.sync, cp.async.bulk.tensor...
└───────────┬──────────────┘
            │ ptxas（离线编译）或驱动内 JIT 编译
            ▼
┌─────────────────────────┐
│      SASS (机器码)         │   真正在 SM 上执行的二进制指令：LDG, HMMA, LDGSTS, UTCGEN5.MMA...
└─────────────────────────┘
```

关键认知：**同一个 C++ API 调用，在不同架构上可能被编译成完全不同的 PTX/SASS 指令序列**（比如 `cuda::memcpy_async` 在 Ampere 上会生成 `cp.async` 指令，而如果编译目标是 Hopper 且数据满足条件，编译器/库可能优先选择走 TMA 路径）。API 的意义正是在于**屏蔽这种因架构而异的底层差异，让同一份高层代码可以跨代际移植**，同时性能敏感的开发者仍然可以通过内联 PTX 或架构专属 intrinsic 精确控制底层生成的指令。

## 7.2 CUDA Runtime API vs CUDA Driver API

- **Runtime API**（`cudaMalloc`、`cudaMemcpy`、`kernel<<<>>>`、`cudaStreamCreate`……）：更高层、更易用，隐式管理 Context（每个 Device 一个默认 Context，随第一次 CUDA 调用自动创建），是绝大多数应用直接使用的接口。
- **Driver API**（`cuMemAlloc`、`cuLaunchKernel`、`cuModuleLoad`、`cuTensorMapEncodeTiled`……）：更底层、更细粒度地控制 Context、Module（对应编译好的 cubin/PTX）、显式的 Kernel 参数打包与启动。第三部分讲的 TMA 描述符构建（`cuTensorMapEncodeTiled`）就必须通过 Driver API，因为它是 Runtime API 尚未完全封装的底层能力。
- **关系**：Runtime API 内部就是基于 Driver API 实现的——`nvcc` 编译出的 host 端代码里，`kernel<<<>>>()` 语法糖最终会展开成一系列 Runtime API 调用（`cudaLaunchKernel`），而 Runtime 库本身再调用 Driver API 真正把命令提交给 GPU。两者可以在同一个程序里混用，但要注意 Context 管理语义上的细微差异。

## 7.3 Stream 与 Graph

Stream 已在第二部分讲过语义。这里补充 **CUDA Graph**：

- **Why**：当一个计算流程由很多小 Kernel 组成、且这个流程会被反复执行（如训练循环的每个 step），每次都用 CPU 端逐个 `cudaLaunchKernel` 提交，会产生不可忽视的 **Launch Overhead**（每次 Kernel 启动，CPU 都要经过 Runtime→Driver→提交命令到硬件队列的一整套流程，微秒级但会累积）。
- **What**：CUDA Graph 允许把一整套 Kernel 启动 + 内存操作 + 依赖关系，**预先构建成一个静态的有向无环图**，之后可以用一次 `cudaGraphLaunch` 把整个图提交给 GPU，硬件按照图中编码好的依赖关系自动排布执行顺序，大幅减少 CPU 端重复下发指令的开销。
- **How**：可以用 Stream Capture（`cudaStreamBeginCapture`/`cudaStreamEndCapture`，把一段正常的 Stream 操作记录下来自动转成 Graph）或显式 Graph API 手工构建节点。

## 7.4 Cooperative Groups

**Why 出现**：早期 CUDA 只有两种粒度的同步——隐式的 Warp 锁步（Volta 之后不再安全）和 `__syncthreads()`（整个 Block）。很多算法需要**灵活可组合的线程分组**：Warp 子集、整个 Grid，以及 Hopper 起的 **Thread Block Cluster**。

Cooperative Groups 回答的是：**CUDA API 如何表达硬件协作能力**。执行层级见[第二部分 2.2.4](./part02-cuda-programming-model.md)；DSM 见[第三部分 3.10](./part03-cuda-memory/08-cluster-dsm-tmem.md)；barrier 语义见[第十三部分](./part13-synchronization-handbook/01-execution-barriers.md)。

### 7.4.1 `thread_block`

| API | 作用 |
|---|---|
| `this_thread_block()` | 当前 CTA 的 group |
| `thread_rank()` | Block 内线性线程序号 |
| `sync()` / `block.sync()` | CTA 会合（通常映射 `__syncthreads` / `bar.sync`） |
| `group_dim()` / `size()` | Block 维度与线程数 |

### 7.4.2 `thread_block_tile` 与 warp 级协作

| API | 作用 |
|---|---|
| `tiled_partition<N>(parent)` | 把 parent 切成大小 N 的静态 tile（N 为 2 的幂，且常 ≤32） |
| `tile.sync()` | tile 内会合 |
| shuffle / meta-group 辅助 | 把手工 Warp Mask 工作交给库 |

适合 Warp 内 reduce/scan；不等价于 CTA/global memory fence。底层常见 `bar.warp.sync` / `shfl.sync` 等，见 Part 13。

### 7.4.3 `coalesced_group`

捕获“当前实际执行到这一行的线程集合”（Divergence 后可能只是 Warp 子集）。用于分支内正确的组内操作；`__activemask()` 只是快照，不自动成为稳定算法 mask。

### 7.4.4 `grid_group`

| API | 作用 |
|---|---|
| `this_grid()` | 整个 Grid |
| `grid.sync()` | 跨 Block 全局会合 |

依赖 **Cooperative Launch**（`cudaLaunchCooperativeKernel`）：全部 Block 必须能同时驻留，否则会出现“已运行 Block 等待尚未调度 Block”的死锁。Grid 尺寸受设备最大驻留 Block 数约束；是 Persistent Kernel（第十一部分）等技巧的基础，**不是**默认跨 Block 通信模型。

### 7.4.5 `cluster_group`（Hopper+）

Cluster 是介于 CTA 与 Grid 之间的协作范围。必须先以 `__cluster_dims__` 或 `cudaLaunchAttributeClusterDimension` launch，否则不应假定下列 API 可用。

| API | 作用 | 归属 |
|---|---|---|
| `this_cluster()` | 当前 Thread Block Cluster | 执行分组 |
| `block_rank()` | 当前 CTA 在 cluster 内的序号 | 执行分组 |
| `num_blocks()` / `dim_blocks()` | cluster 内 CTA 数量 / 维度 | 执行分组 |
| `cluster.sync()` | cluster 范围 execution barrier | **同步** → Part 13 |
| `map_shared_rank(ptr, rank)` | 把本地 shared 指针映射到 peer CTA | **DSM 寻址** → Part 03 |

```cuda
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(int* out) {
  cg::cluster_group cluster = cg::this_cluster();
  extern __shared__ int smem[];

  if (threadIdx.x == 0)
    smem[0] = static_cast<int>(cluster.block_rank()) + 10;

  cluster.sync(); // 会合 + 建立 DSM 交接所需可见性

  if (cluster.block_rank() == 0 && threadIdx.x == 0) {
    int* remote = cluster.map_shared_rank(smem, 1);
    out[0] = remote[0]; // 期望读到 rank1 写入的 11
  }
  cluster.sync(); // 释放 remote 访问后再让 owner 退出
}
```

常见混淆：

| 误用 | 正解 |
|---|---|
| 把 `map_shared_rank` 当成 sync | 它只翻译地址；会合用 `cluster.sync()` |
| 把 `cluster.sync()` 当成 `grid.sync()` | 只覆盖当前 cluster，不覆盖整个 Grid |
| 在非 cluster launch 上调用 `this_cluster()` | 未定义/不可用；先查 `cudaDevAttrClusterLaunch` |
| 用 `block_rank` 代替 `blockIdx` | rank 是 cluster 局部序号；Grid 坐标仍用 `blockIdx` |

可运行示例：[Part 13 `case03_cluster_sync.cu`](../src/part13-synchronization-handbook/01-execution-barriers/case03_cluster_sync.cu)。

### 7.4.6 `memcpy_async`（Group 封装）

`cooperative_groups::memcpy_async` 是对第三部分 `cp.async` / pipeline 路径的 C++ 封装，以 Group 为单位发起集体异步拷贝。完成仍属 async completion，不等于 `block.sync()` / `cluster.sync()`。

## 7.5 `cuda::pipeline` 与 `cuda::barrier`（`<cuda/pipeline>`, `<cuda/barrier>`）

这是对第四部分讲的"生产者-消费者流水线"和第六部分讲的 `mbarrier` 的标准 C++ 封装（来自 libcu++），核心类型：

- **`cuda::barrier<cuda::thread_scope_block>`**：C++ 标准 `std::barrier` 风格的到达-等待屏障，`thread_scope` 模板参数指定作用域（Block/Device/System），底层根据作用域映射到不同强度的同步指令。
- **`cuda::pipeline<cuda::thread_scope_block>`**：封装"发起异步拷贝（`producer_commit`）→ 等待完成（`consumer_wait`）→ 释放缓冲区（`consumer_release`）"的完整协议，并支持指定流水线级数（stage 数），直接对应第四部分的 Double/Triple Buffer 概念——用 `cuda::pipeline` 写多级流水线的 GEMM，不需要手写底层的 `cp.async.commit_group`/`wait_group` 序列。

一个典型用法骨架（示意）：

```cpp
#include <cuda/pipeline>

__global__ void gemm_kernel(...) {
    __shared__ float buf[STAGES][TILE_SIZE];
    auto pipe = cuda::make_pipeline();

    for (int stage = 0; stage < STAGES; ++stage) {
        pipe.producer_acquire();
        cuda::memcpy_async(buf[stage], &global_ptr[...], sizeof(...), pipe);
        pipe.producer_commit();
    }

    for (int i = 0; i < num_tiles; ++i) {
        pipe.consumer_wait();          // 等待当前 stage 数据就绪
        // ... 用 buf[i % STAGES] 做计算 ...
        pipe.consumer_release();       // 释放这个 stage，允许下一次搬运复用这块缓冲区
        if (i + STAGES < num_tiles) {
            pipe.producer_acquire();
            cuda::memcpy_async(buf[(i + STAGES) % STAGES], &global_ptr[...], sizeof(...), pipe);
            pipe.producer_commit();
        }
    }
}
```

## 7.6 WMMA API（`<mma.h>`）

对应第五部分 Volta 引入的 Warp 级 Tensor Core 操作：

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```

WMMA 是**跨架构可移植**的最高层 Tensor Core 接口——同一份代码在 Volta/Turing/Ampere 上都能编译运行（底层 `ptxas` 会生成对应架构的 `mma.sync` 变体），但**不支持 Hopper 的 WGMMA 和 Blackwell 的 `tcgen05`**这类需要 Warp Group/单线程发起、结果落地 TMEM 的新范式——这些必须通过内联 PTX、CUTLASS/CUTE 抽象，或更底层的架构专属 API 才能使用，这是"高层可移植 API 的抽象能力，天然滞后于最新硬件特性"这一现象的直接例子，也是为什么工业级高性能库（CUTLASS）几乎总是紧跟新架构、用内联 PTX 或 `cutlass::arch` 命名空间下的专属封装，而不是等待 WMMA 之类的通用 API 更新。

WMMA、`mma.sync`、WGMMA、数据中心 `tcgen05` 与消费级 `sm_120` block-scale 路径的边界，见[第五部分 Tensor Core 指令手册](./part05-tensor-core-handbook/00-overview.md)。

## 7.7 从 API 到 PTX 到 SASS：一个可以亲自验证的完整例子

理解三层关系最好的方式是亲自查看编译产物：

```bash
# 生成 PTX（保留人类可读的中间表示）
nvcc -arch=sm_90a -ptx kernel.cu -o kernel.ptx

# 生成最终的 SASS（先编译出 cubin，再反汇编）
nvcc -arch=sm_90a -cubin kernel.cu -o kernel.cubin
cuobjdump -sass kernel.cubin
```

对一个使用 `cuda::memcpy_async` 的简单 kernel，满足架构、对齐和 API 前提时，PTX **可能**出现 `cp.async.cg.shared.global` 及配套 group 指令，Ampere SASS 常见 `LDGSTS`。对 WMMA，PTX 可出现 `wmma.*`，也可能被编译器降低为其它相关 MMA 形式；目标 SASS 常见 `HMMA` 系列。应将这些视为指定 Toolkit/target 下的实测结果，而非稳定的一一映射；完整验证流程见第五部分 5.7 节。

## 7.8 小结

这一部分建立的核心认知：**CUDA C++ API 不是与硬件割裂的抽象层，但也不是稳定的一对一“语法糖”映射**——高层调用可能被内联、融合、删除或选择不同的 PTX/SASS 路径。性能优化应在指定 Toolkit、编译选项和目标 GPU 下检查“这行 C++ 最终生成了什么，是否用到了期望硬件能力（例如是否真的走 TMA，而非退化成 `ld.global`）”。下一部分，我们正式进入编译器内部，看 `nvcc`/`ptxas` 如何完成这条链路的转换。
