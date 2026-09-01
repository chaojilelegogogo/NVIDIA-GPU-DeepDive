# 13.1 Execution Barrier Cases

| Case | 目的 | 运行时结果 | PTX/SASS 重点 |
|---|---|---|---|
| 01 CTA barrier/reduce | 比较会合与会合+谓词规约 | sum=2016、count=32、and/or=1 | `bar.sync`、`bar.red.*` |
| 02 Warp sync | warp shared-memory handoff | lane0 读到 lane31 的 31 | `bar.warp.sync`、`WARPSYNC` |
| 03 Cluster sync | cluster launch + DSM remote read | rank0 读到 rank1 的 11 | cluster barrier/DSM 指令 |
| 04 Grid sync | cooperative grid execution barrier | block1 读到 77 | cooperative-grid lowering |
| 05 bar.arrive | arrive 不阻塞 + sync 等待配对 | waiting warp 读到 42 | `bar.arrive` / `bar.sync` |

```bash
cmake --build build --target part13_execution_barriers_inspect -j
ctest --test-dir build -R part13_exec --output-on-failure
```

Cluster/grid case 会先查询设备能力，不支持时打印 `skipped` 并正常返回。跳过不表示普通 kernel 可以软件模拟相同语义。
