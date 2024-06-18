set(LLVM_LINK_COMPONENTS Core Support)

# CMake will put the safe-c executable in LLVM_RUNTIME_OUTPUT_INTDIR by default
# since we add the executable with add_clang_executable(). By setting
# LLVM_RUNTIME_OUTPUT_INTDIR, we can tell Clang to put this executable in this
# directory's corresponding build directory instead. This allows us to build
# safe-c for multiple configuration types for multi-configuration generators
# without them overwriting each other.
set(MACRONI_ORIGINAL_LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

# Use a generator expression to set the build directory in the way CMake would.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
add_clang_executable(safe-c SafeC.cpp)

# Restore the original LLVM_RUNTIME_OUTPUT_INTDIR value.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${MACRONI_ORIGINAL_LLVM_RUNTIME_OUTPUT_INTDIR})

target_link_libraries(safe-c PRIVATE macroni_settings safety_translation_api
                                     safety_consumer safety_matchers MLIRSafety)
mlir_check_link_libraries(safe-c)
