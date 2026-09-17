# 3.8 Ampere `cp.async`：小粒度异步 copy 手册

## 3.8.1 它准确地做了什么

非 bulk `cp.async` 从 global 读取 4/8/16B，并异步写入 shared：

```ptx
cp.async.ca.shared.global [dst_smem], [src_gmem], 16;
cp.async.cg.shared.global [dst_smem], [src_gmem], 16;
```

- `.ca`：cache at all levels 的策略；
- `.cg`：cache at global level；该变体要求 16B copy；
- 数据不需要占用程序可见的中转寄存器；
- 指令对发起线程异步，但仍由线程/warp 发射；
- 一条指令只描述一个很小的片段，搬整 tile 要多个 lane、多轮发起。

## 3.8.2 `src-size` 与 zero fill

PTX 可指定实际有效源字节数，小于 `cp-size` 的尾部在 shared 中补零：

```ptx
// 目标写 16B，源仅有 12B 有效，剩余 4B 清零
cp.async.ca.shared.global [dst], [src], 16, 12;
```

这适合尾块，能减少分支。`src-size` 必须满足该指令的合法范围；不要把它理解成任意 OOB 读取许可，源地址和有效范围仍须合法。

## 3.8.3 group completion

```ptx
cp.async.cg.shared.global [dst0], [src0], 16;
cp.async.cg.shared.global [dst1], [src1], 16;
cp.async.commit_group;

// 发起下一组或计算上一 tile

cp.async.wait_group 0;
```

语义要点：

- `commit_group` 把当前线程此前未提交的 `cp.async` 组成一个 group；
- group 内操作彼此没有额外顺序保证；
- group 按提交顺序完成；
- `wait_group N` 等到“最多还有 N 个较新的/未完成 group”所规定的状态，不是等待编号 N；
- `wait_all` 等价于提交当前 pending 操作并等待全部完成的便捷形式。

group 状态是**per-thread** 的。warp 中 lane 若在不同控制流中发起/commit/wait 不匹配，很容易产生错误或 pipeline entanglement。

## 3.8.4 为什么 `wait_group` 后常还要 CTA 同步

假设 lane 0 搬的数据会被 lane 17 读取：

```text
lane 0 的 async copy 完成
        ≠
lane 17 已经被协议允许读取且所有线程已到消费点
```

常见安全模式是所有生产 lane 等待自己的 copy 完成，再通过 `__syncthreads()` 交接给 CTA 消费者。更复杂 producer/consumer 可用 `cuda::pipeline` 或 barrier，但必须明确参与范围。完整 memory ordering 见同步手册。

## 3.8.5 CUDA C++：`cuda::memcpy_async` + pipeline

概念示例：

```cuda
#include <cuda/pipeline>

template<int BLOCK, int STAGES>
__global__ void kernel(const float* gmem, float* out, int n) {
  extern __shared__ float smem[];
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope_block, STAGES> state;

  auto block = cooperative_groups::this_thread_block();
  auto pipe = cuda::make_pipeline(block, &state);

  for (int tile = 0; tile < num_tiles; ++tile) {
    pipe.producer_acquire();

    int g = tile * BLOCK + threadIdx.x;
    cuda::memcpy_async(
        &smem[(tile % STAGES) * BLOCK + threadIdx.x],
        &gmem[g],
        cuda::aligned_size_t<16>(16),
        pipe);

    pipe.producer_commit();
    pipe.consumer_wait();

    consume(smem + (tile % STAGES) * BLOCK);

    pipe.consumer_release();
  }
}
```

这是协议骨架，不是可直接用于任意 `BLOCK/n` 的完整 kernel：每线程 16B 会改变索引单位，尾块也要单独处理。真正使用时要保证源、目标和 copy size 的对齐承诺真实成立。

## 3.8.6 对齐为何决定是否走硬件路径

`cuda::memcpy_async` 是 C++ 抽象，不保证编译器必然生成 `cp.async`。硬件加速通常要求：

- source 在 global、destination 在 shared；
- size 和地址满足指令约束；
- 编译器能够证明对齐。

可用 `cuda::aligned_size_t<N>` 或 `cuda::aligned_size_t` 风格的 shape 告知编译器，但错误承诺会导致未定义行为。无法静态证明时，库可能发出运行时检查或回退到普通 copy。

## 3.8.7 正确的双缓冲时间线

```text
时间 →

stage 0: issue tile 0 ───────────── ready ─ compute tile 0
stage 1:              issue tile 1 ───────────── ready ─ compute tile 1
stage 0:                                      issue tile 2 ─────── ...
```

关键是把 wait 放在“首次消费该 stage”之前，而不是紧跟 commit：

```text
错误：issue → commit → wait all → compute
正确：issue next → compute current → wait next → swap
```

stage 数不是越多越好。增加 stage 会占用更多 shared memory，可能降低 occupancy；通常从 2/3 stage 实测。

## 3.8.8 Warp entanglement

`cuda::pipeline` 在 Ampere 的底层 batch 序列可能由 warp 共享。若 warp 严重分歧地执行 commit/wait：

- 实际 sequence 可能比单线程感知的 sequence 更快推进；
- wait 可能等待比预期更多的 batch；
- barrier arrive 可能被重复更新。

因此在 producer commit、consumer wait 等点保持 warp convergence，必要时先 `__syncwarp()`。

## 3.8.9 常见错误

| 症状 | 原因 |
|---|---|
| 结果偶发旧值 | 只 wait 了部分 lane 的 group，缺少跨线程交接 |
| 没有性能提升 | commit 后立即 wait；tile 太小；没有计算覆盖延迟 |
| 编译后仍是 LDG+STS | 地址空间/对齐/size 不满足，或目标低于 `sm_80` |
| 尾块越界 | 把 zero-fill 的 `src-size` 误当成可读取任意越界地址 |
| pipeline 等待异常 | 分歧路径导致 commit/wait 协议不一致 |
| occupancy 大降 | stage buffer 占用过多 shared memory |

## 3.8.10 验证

```bash
nvcc -arch=sm_80 -lineinfo kernel.cu -o kernel
cuobjdump --dump-ptx kernel
cuobjdump --dump-sass kernel
```

Ampere SASS 中常见 `LDGSTS` 线索，但名字和具体 lowering 以 Toolkit/目标实测为准。再用 Nsight Compute 对比：

- register 数量；
- global/shared throughput；
- long scoreboard stall；
- shared bank conflict；
- active warps 与 shared-memory occupancy 限制。

`commit_group/wait_group`、warp entanglement 和跨线程交接的完整同步语义见[第十三部分 Async Pipeline Synchronization](../part13-synchronization-handbook/03-async-pipelines.md)；流水线调度与 latency hiding 见[第四部分执行流水线](../../vol1-gpu-hardware/part04-execution-pipeline.md)。

下一篇：[Hopper bulk copy 与 TMA](05-hopper-bulk-and-tma.md)。
