include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AddCudaCase.cmake")

add_part13_cuda_case(part13_exec_case01_cta_barrier_reduce
  case01_cta_barrier_reduce.cu)
add_part13_cuda_case(part13_exec_case02_warp_sync
  case02_warp_sync.cu)
add_part13_cuda_case(part13_exec_case03_cluster_sync
  case03_cluster_sync.cu)
add_part13_cuda_case(part13_exec_case04_grid_sync
  case04_grid_sync.cu)
add_part13_cuda_case(part13_exec_case05_bar_arrive
  case05_bar_arrive.cu)

add_custom_target(part13_execution_barriers_inspect
  DEPENDS
    part13_exec_case01_cta_barrier_reduce_inspect
    part13_exec_case02_warp_sync_inspect
    part13_exec_case03_cluster_sync_inspect
    part13_exec_case04_grid_sync_inspect
    part13_exec_case05_bar_arrive_inspect
)
