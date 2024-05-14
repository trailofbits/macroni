# This function prevents in-source builds.
function(_prevent_in_source_builds)
  # Add message context for more informative output with --log-context option.
  # Note that we don't need to pop this addition to the message context at the
  # end of this function since we are appending to a function-local copy of
  # CMAKE_MESSAGE_CONTEXT.
  list(APPEND CMAKE_MESSAGE_CONTEXT "preventInSourceBuilds")

  # Resolve real paths.
  file(REAL_PATH "${CMAKE_SOURCE_DIR}" SOURCE_DIR)
  file(REAL_PATH "${CMAKE_BINARY_DIR}" BINARY_DIR)

  # Check if we are attempting an in-source build.
  if("${SOURCE_DIR}" STREQUAL "${BINARY_DIR}")
    message(NOTICE "In-source build detected")
    message(NOTICE "In-source builds are disabled")
    message(NOTICE
            "Please create a separate build directory and run cmake from there")
    message(FATAL_ERROR "Quitting configuration")
  endif()

endfunction()

_prevent_in_source_builds()
