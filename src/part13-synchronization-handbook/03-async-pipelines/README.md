# 13.3 Async Pipeline Cases

| Case | 目的 | 重点 |
|---|---|---|
| 01 cp.async pipeline | stage acquire/commit/wait/release | `cp.async` group；是否真正生成取决于目标和对齐 |
| 02 mbarrier phases | barrier token、phase 复用 | `mbarrier.init/arrive/try_wait` |
| 03 TMA transaction | arrival + transaction bytes | `cp.async.bulk.tensor` + mbarrier completion |

```bash
cmake --build build --target part13_async_pipelines_inspect -j
ctest --test-dir build -R part13_async --output-on-failure
```

Case 03 需要 Hopper+；不支持时安全跳过。TMA 测试使用的公开 PTX/C++ wrapper 对应 `cp.async.bulk.tensor`，不是虚构的 `tma.async.load` 指令。CUDA 13 / Thor 上 1D `cuTensorMapEncodeTiled` 需显式传入 `globalStrides`（元素字节步长），不能传 `nullptr`。
