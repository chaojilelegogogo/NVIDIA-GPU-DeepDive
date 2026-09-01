# Part 13 Synchronization Cases

| 文档 | Case 目录 | 覆盖范围 |
|---|---|---|
| 13.1 Execution Barrier | `01-execution-barriers/` | Warp、CTA（含 arrive/red）、Cluster、Grid |
| 13.2 Memory Ordering | `02-memory-ordering/` | order、scope、fence、ordered atomic |
| 13.3 Async Pipeline | `03-async-pipelines/` | cp.async、mbarrier phase、TMA transaction |
| 13.4 Tensor Synchronization | `04-tensor-synchronization/` | mma.sync 基线；WGMMA/tcgen05 目标限制 |
| 13.5 Collectives/Atomics | `05-collectives-and-atomics/` | shfl/vote/match、atom vs red |

## 构建全部可运行案例

```bash
cmake -S . -B build -DNVGPU_CUDA_ARCH=native
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 生成全部 PTX/CUBIN/SASS

```bash
cmake --build build --target part13_sync_inspect -j
```

每个目录还提供更小的 `*_inspect` target。Cluster、TMA、cooperative grid 等 case 会查询运行时能力；不支持时明确打印 `skipped`，不会假装普通 barrier 等价。
