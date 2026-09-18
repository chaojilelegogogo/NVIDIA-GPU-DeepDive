# Part 10 SASS Cases

| Case | 目的 | 观察重点 |
|---|---|---|
| 01 FFMA | 最小 fused multiply-add | CUDA `__fmaf_rn` → PTX `fma.rn.f32` → SASS `FFMA` |

本目录面向 Blackwell（Thor / `sm_110` 等）本机验证；用 `-DNVGPU_CUDA_ARCH=native` 即可。

```bash
cmake -S . -B build -DNVGPU_CUDA_ARCH=native
cmake --build build -j --target part10_case01_ffma part09_sass_inspect
ctest --test-dir build -R part09_ --output-on-failure

# 产物
#   build/.../artifacts/part10_case01_ffma.ptx
#   build/.../artifacts/part10_case01_ffma.sass
```

对照文档：[第十部分 SASS](../../docs/vol2-cuda-software/part10-sass.md) §10.1.1。
