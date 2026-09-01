if(NOT DEFINED TOOL OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "TOOL, INPUT and OUTPUT must be provided")
endif()

execute_process(
  COMMAND "${TOOL}" --dump-sass "${INPUT}"
  RESULT_VARIABLE result
  OUTPUT_FILE "${OUTPUT}"
  ERROR_VARIABLE error_output
)

if(NOT result EQUAL 0)
  message(FATAL_ERROR "cuobjdump failed (${result}): ${error_output}")
endif()
