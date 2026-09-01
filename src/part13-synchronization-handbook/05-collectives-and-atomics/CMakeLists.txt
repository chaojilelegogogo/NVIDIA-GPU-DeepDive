include("${CMAKE_CURRENT_LIST_DIR}/../cmake/AddCudaCase.cmake")

add_part13_cuda_case(part13_collective_case01_warp_collectives
  case01_warp_collectives.cu)
add_part13_cuda_case(part13_collective_case02_atom_vs_red
  case02_atom_vs_red.cu)

add_custom_target(part13_collectives_atomics_inspect
  DEPENDS
    part13_collective_case01_warp_collectives_inspect
    part13_collective_case02_atom_vs_red_inspect
)
