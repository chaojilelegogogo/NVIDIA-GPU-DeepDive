# 13.4 Tensor Synchronization Cases

当前提供 `case01_mma_sync.cu` 作为所有设备可理解的同步式 Tensor Core 基线：

```bash
cmake --build build --target part13_tensor_synchronization_inspect
ctest --test-dir build -R part13_tensor --output-on-failure
```

## 为什么没有伪造 WGMMA/tcgen05 最小 case

WGMMA 和 `tcgen05` 不是只写 mnemonic 就能成立：

- WGMMA 要求合法的 128-thread warpgroup、matrix descriptor、operand layout 和 fence/commit/wait；
- `tcgen05` 要求数据中心 Blackwell architecture-specific target、TMEM allocation、CTA group 和完整 required synchronization；
- 消费级 `sm_120` 不支持 TMEM/`tcgen05`。

因此目标相关案例应在 Part 5 的 CuTe/CUTLASS GEMM 基础上提取，而不是放置能编译但 operand/参与协议错误的 inline PTX。这里先用 `mma.sync` 明确“同步 Tensor 指令结果依赖”与 async Tensor completion 的分界。
