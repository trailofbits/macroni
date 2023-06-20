// Copyright (c) 2023-present, Trail of Bits, Inc.


#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Dialect/Macroni/MacroniOps.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/FunctionImplementation.h>
#include <llvm/Support/ErrorHandling.h>

namespace macroni::macroni {

    void MacroExpansionStmt::build(
        mlir::OpBuilder &odsBuilder,
        mlir::OperationState &odsState,
        std::optional<llvm::function_ref<void(mlir::OpBuilder &,
                                              mlir::Location) >>
        expansionBuilder,
        mlir::StringAttr macroName,
        mlir::ArrayAttr parameterNames,
        mlir::BoolAttr functionLike) {
        mlir::OpBuilder::InsertionGuard guard(odsBuilder);
        odsState.addAttribute("macroName", macroName);
        odsState.addAttribute("parameterNames", parameterNames);
        odsState.addAttribute("functionLike", functionLike);
        auto reg = odsState.addRegion();
        if (expansionBuilder.has_value()) {
            odsBuilder.createBlock(reg);
            expansionBuilder.value()(odsBuilder, odsState.location);
        }
    }
}

#define GET_OP_CLASSES
#include "macroni/Dialect/Macroni/Macroni.cpp.inc"