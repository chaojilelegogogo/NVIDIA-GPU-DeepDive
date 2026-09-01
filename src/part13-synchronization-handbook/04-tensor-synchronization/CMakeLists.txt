include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AddCudaCase.cmake")

add_part13_cuda_case(part13_tensor_case01_mma_sync
  case01_mma_sync.cu)

add_custom_target(part13_tensor_synchronization_inspect
  DEPENDS part13_tensor_case01_mma_sync_inspect)
