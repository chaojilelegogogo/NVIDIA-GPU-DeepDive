set(_artifact_dir "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
set(_dump_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DumpSass.cmake")

function(add_part09_sass_case case_number case_name)
  set(target "part09_case${case_number}_${case_name}")
  set(source "${CMAKE_CURRENT_SOURCE_DIR}/case${case_number}_${case_name}.cu")
  set(ptx "${_artifact_dir}/${target}.ptx")
  set(cubin "${_artifact_dir}/${target}.cubin")
  set(sass "${_artifact_dir}/${target}.sass")

  add_executable("${target}" "${source}")
  target_include_directories("${target}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries("${target}" PRIVATE CUDA::cudart)
  target_compile_options("${target}" PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
  add_test(NAME "${target}" COMMAND "${target}")

  add_custom_command(
    OUTPUT "${ptx}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_artifact_dir}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            -std=c++17 -arch=${NVGPU_CUDA_ARCH} -lineinfo -ptx
            -I"${CMAKE_CURRENT_SOURCE_DIR}"
            "${source}" -o "${ptx}"
    DEPENDS "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/common.cuh"
    VERBATIM
    COMMENT "Generating PTX for ${target}"
  )

  add_custom_command(
    OUTPUT "${cubin}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_artifact_dir}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            -std=c++17 -arch=${NVGPU_CUDA_ARCH} -lineinfo -cubin
            -I"${CMAKE_CURRENT_SOURCE_DIR}"
            "${source}" -o "${cubin}"
    DEPENDS "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/common.cuh"
    VERBATIM
    COMMENT "Generating CUBIN for ${target}"
  )

  add_custom_command(
    OUTPUT "${sass}"
    COMMAND "${CMAKE_COMMAND}"
            -DTOOL=${CUDAToolkit_BIN_DIR}/cuobjdump
            -DINPUT=${cubin}
            -DOUTPUT=${sass}
            -P "${_dump_script}"
    DEPENDS "${cubin}" "${_dump_script}"
    VERBATIM
    COMMENT "Disassembling SASS for ${target}"
  )

  add_custom_target("${target}_inspect" DEPENDS "${ptx}" "${sass}")
endfunction()

add_part09_sass_case(01 ffma)

add_custom_target(part09_sass_inspect
  DEPENDS part10_case01_ffma_inspect
)
