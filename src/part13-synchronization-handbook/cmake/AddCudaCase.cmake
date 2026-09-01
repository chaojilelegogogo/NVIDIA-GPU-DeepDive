include(CMakeParseArguments)

function(add_part13_cuda_case target source)
  cmake_parse_arguments(ARG "NO_TEST" "" "LIBRARIES" ${ARGN})

  get_filename_component(source_abs "${source}" ABSOLUTE
                         BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(artifact_dir "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
  set(ptx "${artifact_dir}/${target}.ptx")
  set(cubin "${artifact_dir}/${target}.cubin")
  set(sass "${artifact_dir}/${target}.sass")
  set(dump_script
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DumpSass.cmake")

  add_executable("${target}" "${source_abs}")
  target_include_directories("${target}" PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.."
  )
  target_link_libraries("${target}" PRIVATE CUDA::cudart ${ARG_LIBRARIES})
  target_compile_options("${target}" PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  if(NOT ARG_NO_TEST)
    add_test(NAME "${target}" COMMAND "${target}")
  endif()

  add_custom_command(
    OUTPUT "${ptx}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${artifact_dir}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            -std=c++17 -arch=${NVGPU_CUDA_ARCH} -lineinfo -ptx
            -I"${CMAKE_CURRENT_SOURCE_DIR}"
            -I"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.."
            "${source_abs}" -o "${ptx}"
    DEPENDS "${source_abs}"
    VERBATIM
    COMMENT "Generating PTX for ${target}"
  )

  add_custom_command(
    OUTPUT "${cubin}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${artifact_dir}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            -std=c++17 -arch=${NVGPU_CUDA_ARCH} -lineinfo -cubin
            -I"${CMAKE_CURRENT_SOURCE_DIR}"
            -I"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.."
            "${source_abs}" -o "${cubin}"
    DEPENDS "${source_abs}"
    VERBATIM
    COMMENT "Generating CUBIN for ${target}"
  )

  add_custom_command(
    OUTPUT "${sass}"
    COMMAND "${CMAKE_COMMAND}"
            -DTOOL=${CUDAToolkit_BIN_DIR}/cuobjdump
            -DINPUT=${cubin}
            -DOUTPUT=${sass}
            -P "${dump_script}"
    DEPENDS "${cubin}" "${dump_script}"
    VERBATIM
    COMMENT "Disassembling SASS for ${target}"
  )

  add_custom_target("${target}_inspect" DEPENDS "${ptx}" "${sass}")
endfunction()
