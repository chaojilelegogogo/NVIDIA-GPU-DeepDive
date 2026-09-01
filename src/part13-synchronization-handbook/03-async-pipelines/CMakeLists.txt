include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AddCudaCase.cmake")

add_part13_cuda_case(part13_async_case01_cp_async_pipeline
  case01_cp_async_pipeline.cu)
add_part13_cuda_case(part13_async_case02_mbarrier_phases
  case02_mbarrier_phases.cu)
add_part13_cuda_case(part13_async_case03_tma_transaction
  case03_tma_transaction.cu
  LIBRARIES CUDA::cuda_driver)

add_custom_target(part13_async_pipelines_inspect
  DEPENDS
    part13_async_case01_cp_async_pipeline_inspect
    part13_async_case02_mbarrier_phases_inspect
    part13_async_case03_tma_transaction_inspect
)
