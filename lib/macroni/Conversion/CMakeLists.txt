add_library(macroni_conversion
    MacroniRewriters.cpp
)

target_link_libraries(macroni_conversion
  PRIVATE
    MLIRLowLevel
    MLIRHighLevel
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRExecutionEngine
    MLIRLLVMCommonConversion
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration

    macroni::settings
)
add_library(macroni::conversion ALIAS macroni_conversion)
