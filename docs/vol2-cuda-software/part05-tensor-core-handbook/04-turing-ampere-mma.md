# 5.4 Part 3：Turing/Ampere —— `mma.sync`、`ldmatrix` 与 `cp.async`

## 5.4.1 为什么 WMMA 不够

WMMA 很适合学习，但隐藏了 instruction shape、寄存器 tuple 和 Shared Memory layout。高性能 CUTLASS/CuTe kernel 需要准确控制：

- 一条 MMA 用什么 `mMnNkK` shape 和数据类型；
- 每 lane 负责哪些 A/B/C/D register；
- Shared tile 如何排列以避免 bank conflict；
- 数据搬运何时开始、何时才真正等待。

Turing 扩展 INT8/INT4 等推理路径；Ampere 增加 TF32/BF16 和更丰富的 `mma.sync` 形状。这里的核心仍是 Warp 寄存器 MMA。

## 5.4.2 `mma.sync.aligned` 到底做什么

以常见 Ampere FP16 → FP32 accumulator 路径为例：

```ptx
// 示意：operand register 个数与类型必须严格按 PTX ISA。
mma.sync.aligned.m16n8k16.row.col.f16.f16.f32.f32
  {d0, d1, d2, d3},
  {a0, a1},
  {b0, b1},
  {c0, c1, c2, c3};
```

逐段理解：

| 片段 | 含义 |
|---|---|
| `mma.sync` | Warp 协作矩阵 FMA；不是 CTA barrier |
| `aligned` | 参与 Warp 必须以对齐的协作方式执行 |
| `m16n8k16` | A 是 16×16，B 是 16×8，输出/累加是 16×8 |
| `row.col` | A/B 的逻辑 layout 约定 |
| `f16.f16.f32.f32` | A 类型、B 类型、C accumulator 类型、D 输出类型 |
| `{a...}` 等 | 每 lane 保存的分布式 fragment 寄存器 tuple |

这一条指令实现 `D=A×B+C`，但 16×8 的 D 元素分散在 Warp 的寄存器中。之后循环不同 K tile，会将 D 再作为 C 输入继续累加；最后才由 epilogue 写回 global memory。

## 5.4.3 `ldmatrix`：Shared Memory → 正确的寄存器 fragment

`mma.sync` 不能把任意普通 load 的结果当作 operand。`ldmatrix` 让 Warp 协作把 Shared Memory tile 分发成规定的寄存器 layout：

```ptx
// 示意：x1/x2/x4、.trans 与地址规则必须匹配目标 MMA。
ldmatrix.sync.aligned.m8n8.x4.shared.b16
  {r0, r1, r2, r3}, [smem_addr];
```

| 原语 | 作用 | 常见 SASS 线索 |
|---|---|---|
| `ld.shared` | 每 lane 线性读取 | `LDS` 等 |
| `ldmatrix` | Warp 协作读取并按 Tensor fragment 排布 | `LDSM` |
| `mma.sync` | 使用分布式 tuple 执行矩阵 FMA | `HMMA` |

正确流程：将 global tile 写入 shared → CTA/warp 正确同步 → `ldmatrix` → `mma.sync`。`.trans` 是指定的 fragment 分配变体，不是通用矩阵转置；Shared layout、padding/swizzle、`ldmatrix` 形状和 MMA layout 必须作为一个整体设计。

## 5.4.4 `cp.async`：为什么 Tensor Core 章节要学习它

Ampere 前的搬运会经过寄存器：

```cuda
half v = global_ptr[i];     // global → register
shared_tile[i] = v;         // register → shared
```

Ampere 可使用 `cp.async` 直接 global → shared：

```ptx
cp.async.cg.shared.global [smem_dst], [gmem_src], 16;
cp.async.commit_group;
// 此处计算上一 tile
cp.async.wait_group 0;  // 真正消费本 tile 前再等待
```

| 指令 | 含义 |
|---|---|
| `cp.async` | 发起一小块 global→shared 异步复制，不经普通寄存器中转 |
| `commit_group` | 将此前发起的 copy 提交为一个 group |
| `wait_group N` | 等待直到最多 N 个较早 group 未完成 |

不要将 `wait_group 0` 紧跟 `commit_group`，否则没有搬运/计算重叠。多 stage GEMM 会预取下一 K tile、计算当前 tile、只在首次 `ldmatrix` 消费时等待当前 tile。

## 5.4.5 一个 Ampere mainloop 的伪代码

```text
预取 stage 0 的 A/B 到 shared，commit
for each K stage:
  wait 当前 stage 的 copy 完成
  必要的 CTA/warp 同步
  ldmatrix 载入 A/B register fragments
  mma.sync 累加到 C registers
  预取未来 stage，commit
写回 accumulator（epilogue）
```

同步语义请结合[第十三部分 13.3](../part13-synchronization-handbook/03-async-pipelines.md)：`cp.async` completion、CTA barrier 和 memory fence 解决的是不同问题。

下一篇：[Hopper 的 TMA/WGMMA](05-hopper-wgmma.md)。
