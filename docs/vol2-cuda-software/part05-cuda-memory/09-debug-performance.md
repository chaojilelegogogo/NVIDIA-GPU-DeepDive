# 5.13 正确性、编译验证与性能诊断

## 5.13.1 先写清目标矩阵

每个实验在文件头记录：

```text
CUDA Toolkit:
PTX ISA:
nvcc -arch:
GPU name / compute capability:
Driver:
Expected path: LDG+STS / cp.async / bulk / TMA / tcgen05
```

这能避免把 `sm_90`、`sm_100a`、`sm_120` 的差异误判成代码 bug。

## 5.13.2 四层验证

### CUDA/CuTe 源码

确认 API、layout、barrier state 与 launch topology。

### PTX

```bash
nvcc -arch=sm_90a -keep -lineinfo kernel.cu
```

搜索：

```text
cp.async
cp.async.bulk
cp.async.bulk.tensor
cp.reduce.async.bulk
mbarrier
wgmma
tcgen05
```

API 名叫 async 不代表一定生成目标 PTX。

### SASS

```bash
cuobjdump --dump-sass a.out
nvdisasm kernel.cubin
```

SASS mnemonic 会随架构/Toolkit 变化，用来确认真实 lowering，不要维护一张永远不变的一一映射表。

### Profile

用 Nsight Compute/Systems 回答：

- copy 与 compute 是否时间重叠；
- DRAM/L2/shared 哪层饱和；
- warp 在等 long scoreboard、barrier 还是 Tensor dependency；
- shared bank conflict 是否降低；
- stage 增加是否因 occupancy 下降而抵消收益。

## 5.13.3 正确性测试矩阵

至少覆盖：

| 维度 | 测试值 |
|---|---|
| tile | 1 个、多个、最后半块 |
| 坐标 | 0、正偏移、合法负起点 |
| shape | 16B 整倍数、带 padding stride |
| OOB | 左/右/上下/多维越界 |
| stage | 1、2、3，循环多次复用 phase |
| CTA | 单 CTA、多 CTA、cluster boundary |
| type | 普通整数/浮点、目标支持的 sub-byte |
| layout | no swizzle 与每个候选 swizzle |

每次与简单 CPU reference 或普通 CUDA load/store baseline 比较。异步 race 常在循环第二轮、尾块或低 occupancy 下才出现。

## 5.13.4 Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./test
compute-sanitizer --tool racecheck ./test
compute-sanitizer --tool synccheck ./test
```

工具未必能完整理解所有最新异步/TMEM 语义，但仍可发现大量地址、barrier participation 与 shared race。工具无报错不构成同步正确性的证明。

## 5.13.5 TensorMap 编码失败清单

`cuTensorMapEncode*` 返回错误时逐项检查：

1. rank 是否符合 API（tiled 为 1–5；im2col 通常为 3–5）；
2. fastest-changing dimension 是否放在 index 0；
3. global base 是否 16B/32B 对齐；
4. `globalStrides` 是否以**字节**为单位并满足 16B/32B 对齐；
5. dimension、box、element stride 是否在范围内；
6. swizzle span 与 inner box bytes 是否匹配；
7. interleave 与 swizzle 组合是否合法；
8. packed U4/U6 的 dim0、box、坐标与 direction 限制；
9. OOB fill 是否适用于 element type；
10. Driver 是否支持用于编译的 Toolkit API 版本。

## 5.13.6 Hang 的系统排查

TMA/mbarrier kernel hang 通常来自：

```text
barrier 未初始化或初始化未对 async proxy 可见
expected arrival 数量错误
expect_tx bytes 错误
有线程未 arrive
等待了旧 phase token
stage 被提前复用
remote mbarrier 地址/CTA rank 错误
cluster 中某 CTA 提前退出
```

排查时把多 stage 降成 1、单 CTA、单 tile；在每个 phase 写 debug state 到独立 global buffer。不要在等待循环中用大量 `printf` 改变调度后误以为修复。

## 5.13.7 “正确但不快”的排查

| 现象 | 可能原因 | 验证 |
|---|---|---|
| TMA 与普通 copy 相同 | tile 太小/复用低 | 扫 tile size |
| 没有 overlap | wait 紧跟 issue | timeline + source inspection |
| producer 气泡 | producer warp 调度不足 | eligible warps、warp specialization |
| stage 越多越慢 | shared 占用降低 occupancy | launch stats |
| swizzle 后更慢 | consumer layout 不匹配或本来无 conflict | shared conflict + 指令数 |
| multicast 无收益 | L2 已命中、fan-out 小、同步开销大 | DRAM/L2 bytes |
| sub-byte 无收益 | unpack/layout/epilogue 成瓶颈 | 分阶段 microbenchmark |
| bulk reduce 慢 | contention/atomic destination 热点 | 改变 destination 分片 |

## 5.13.8 建议 benchmark 方式

1. warm-up；
2. CUDA event 统计多次迭代；
3. 固定时钟/功耗状态条件尽量一致；
4. 输出 median 与分位数，不只取最好一次；
5. 同时报告 effective requested bandwidth 与硬件实际 throughput；
6. 防止编译器消除结果；
7. 将 descriptor 构建排除在 kernel steady-state 时间外，除非业务确实每次重建。

## 5.13.9 Code review 清单

- [ ] `-arch` 与实际 GPU 匹配；
- [ ] descriptor 维度顺序、byte stride 正确；
- [ ] source/destination/size 对齐正确；
- [ ] copy direction 与 completion mechanism 匹配；
- [ ] mbarrier arrival、transaction bytes、phase/token 完整；
- [ ] async completion 与 CTA/cluster 会合没有混淆；
- [ ] generic/shared 与 async/Tensor proxy 间 fence 正确；
- [ ] stage 只在 producer/consumer 都 release 后复用；
- [ ] OOB fill 是算法正确的中性值；
- [ ] swizzle 与消费者 layout 一致；
- [ ] cluster/CTA pair 生命周期正确；
- [ ] sub-byte 和 gather/im2col target restriction 已核对；
- [ ] 生成 PTX/SASS 与预期一致；
- [ ] 有普通路径 baseline 和 profile 证据。

## 5.13.10 参考资料

- [PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C++ Programming Guide：Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)
- [CUDA C++ Programming Guide：TMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
- [CUDA Driver API：Tensor Memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Controlling Data Movement on Ampere](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/)

知识库内继续阅读：[第十部分 SASS](../part10-sass.md)、[第十一部分性能优化](../part11-performance-optimization.md)、[第十二部分 Kernel 源码分析](../part12-kernel-source-analysis.md)。

返回：[第五部分总览](00-overview.md)。
