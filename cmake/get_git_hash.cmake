function(get_git_hash function_output_variable)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    RESULT_VARIABLE result
    OUTPUT_VARIABLE git_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (result)
    message(FATAL_ERROR "Failed to get git hash: ${result}")
  endif()
  set(${function_output_variable} ${git_hash} PARENT_SCOPE)
endfunction()
