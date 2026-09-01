file(REMOVE_RECURSE
  "CMakeFiles/part13_memory_order_inspect"
  "artifacts/part13_case01_plain_vs_release_acquire.cubin"
  "artifacts/part13_case01_plain_vs_release_acquire.ptx"
  "artifacts/part13_case01_plain_vs_release_acquire.sass"
  "artifacts/part13_case02_relaxed_vs_release_rmw.cubin"
  "artifacts/part13_case02_relaxed_vs_release_rmw.ptx"
  "artifacts/part13_case02_relaxed_vs_release_rmw.sass"
  "artifacts/part13_case03_acquire_load.cubin"
  "artifacts/part13_case03_acquire_load.ptx"
  "artifacts/part13_case03_acquire_load.sass"
  "artifacts/part13_case04_acq_rel_lock.cubin"
  "artifacts/part13_case04_acq_rel_lock.ptx"
  "artifacts/part13_case04_acq_rel_lock.sass"
  "artifacts/part13_case05_threadfence_vs_release.cubin"
  "artifacts/part13_case05_threadfence_vs_release.ptx"
  "artifacts/part13_case05_threadfence_vs_release.sass"
  "artifacts/part13_case06_scopes.cubin"
  "artifacts/part13_case06_scopes.ptx"
  "artifacts/part13_case06_scopes.sass"
  "artifacts/part13_case07_seq_cst.cubin"
  "artifacts/part13_case07_seq_cst.ptx"
  "artifacts/part13_case07_seq_cst.sass"
  "artifacts/part13_case08_order_scope_payload.cubin"
  "artifacts/part13_case08_order_scope_payload.ptx"
  "artifacts/part13_case08_order_scope_payload.sass"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/part13_memory_order_inspect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
