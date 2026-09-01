# CMake generated Testfile for 
# Source directory: /mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines
# Build directory: /mnt/nt00098/nvgpu_learn/build/src/part13-synchronization-handbook/03-async-pipelines
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[part13_async_case01_cp_async_pipeline]=] "/mnt/nt00098/nvgpu_learn/build/src/part13-synchronization-handbook/03-async-pipelines/part13_async_case01_cp_async_pipeline")
set_tests_properties([=[part13_async_case01_cp_async_pipeline]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/cmake/AddCudaCase.cmake;26;add_test;/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines/CMakeLists.txt;3;add_part13_cuda_case;/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines/CMakeLists.txt;0;")
add_test([=[part13_async_case02_mbarrier_phases]=] "/mnt/nt00098/nvgpu_learn/build/src/part13-synchronization-handbook/03-async-pipelines/part13_async_case02_mbarrier_phases")
set_tests_properties([=[part13_async_case02_mbarrier_phases]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/cmake/AddCudaCase.cmake;26;add_test;/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines/CMakeLists.txt;5;add_part13_cuda_case;/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines/CMakeLists.txt;0;")
add_test([=[part13_async_case03_tma_transaction]=] "/mnt/nt00098/nvgpu_learn/build/src/part13-synchronization-handbook/03-async-pipelines/part13_async_case03_tma_transaction")
set_tests_properties([=[part13_async_case03_tma_transaction]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/cmake/AddCudaCase.cmake;26;add_test;/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines/CMakeLists.txt;7;add_part13_cuda_case;/mnt/nt00098/nvgpu_learn/src/part13-synchronization-handbook/03-async-pipelines/CMakeLists.txt;0;")
