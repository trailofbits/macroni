add_library(kernel_conversion
    KernelRewriters.cpp
)

target_link_libraries(kernel_conversion
    PRIVATE
        MLIRIR
        MLIRPass
        MLIRTransformUtils
        MLIRExecutionEngine
        MLIRLLVMCommonConversion
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
)
add_library(macroni::kernel_conversion ALIAS kernel_conversion)
