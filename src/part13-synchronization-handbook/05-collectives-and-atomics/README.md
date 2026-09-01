# 13.5 Collective and Atomic Cases

| Case | 目的 | PTX 重点 |
|---|---|---|
| 01 Warp collectives | mask、shuffle、vote、match 的参与协议 | `activemask`、`shfl.sync`、`vote.sync`、`match.sync` |
| 02 Atom vs red | 返回旧值是否被使用对 lowering 的影响 | `atom.*` 与 `red.*` |

```bash
cmake --build build --target part13_collectives_atomics_inspect -j
ctest --test-dir build -R part13_collective --output-on-failure
```

Atomic order/scope 不在这里重复实现；请使用相邻 `02-memory-ordering` 的 8 个 case，对比 relaxed/acquire/release/acq_rel/seq_cst、CTA/GPU/SYS scope，以及 Order×Scope payload 协议。
