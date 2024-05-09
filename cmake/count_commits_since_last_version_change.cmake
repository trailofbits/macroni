# Counts the number of commits made since the last change to the project
# version. Requires that Git already be loaded and the file
# cmake/project_version_details.cmake exists

function(count_commits_since_last_version_change function_output_variable)

  if(NOT Git_FOUND)
    message(FATAL_ERROR "Git not found")
  endif()

  set(project_version_file
      "${PROJECT_SOURCE_DIR}/cmake/project_version_details.cmake")

  if(NOT IS_READABLE ${project_version_file})
    message(FATAL_ERROR "${project_version_file}" does not exist!)
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-list -1 HEAD "${project_version_file}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE last_version_change_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(result)
    message(
      FATAL_ERROR "Failed to get git hash of last version change: ${result}")
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-list --count ${last_version_change_hash}..HEAD
    RESULT_VARIABLE result
    OUTPUT_VARIABLE hash_count
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(result)
    message(
      FATAL_ERROR
        "Failed to count the number of commits since last version change: ${result}"
    )
  endif()

  set(${function_output_variable}
      ${hash_count}
      PARENT_SCOPE)
endfunction()
