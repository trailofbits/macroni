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

    void MacroParameterStmt::build(
        mlir::OpBuilder &odsBuilder,
        mlir::OperationState &odsState,
        std::optional<llvm::function_ref<void(mlir::OpBuilder &,
                                              mlir::Location) >>
        expansionBuilder,
        mlir::StringAttr parameterName) {
        mlir::OpBuilder::InsertionGuard guard(odsBuilder);
        odsState.addAttribute("parameterName", parameterName);
        auto reg = odsState.addRegion();
        if (expansionBuilder.has_value()) {
            odsBuilder.createBlock(reg);
            expansionBuilder.value()(odsBuilder, odsState.location);
        }
    }

    mlir::ParseResult parseMacroParameters(
        mlir::OpAsmParser &parser,
        mlir::StringAttr &macroName,
        mlir::BoolAttr &functionLike,
        mlir::ArrayAttr &parameterNames) {

        mlir::Builder &builder = parser.getBuilder();

        std::string string_result;
        mlir::ParseResult parse_result = parser.parseString(&string_result);
        if (parse_result.succeeded()) {
            macroName = builder.getStringAttr(llvm::Twine(string_result));
        } else {
            return mlir::failure();
        }

        // NOTE(bpp): Use optional LParen or mandatory LParen here? Not sure
        // what the difference is if we check for success either way
        parse_result = parser.parseOptionalLParen();
        if (parse_result.succeeded()) {
            functionLike = builder.getBoolAttr(true);
            llvm::SmallVector<llvm::StringRef, 4> param_names_vec;
            parse_result = parser.parseCommaSeparatedList(
                [&]() {
                    if (parser.parseString(&string_result).succeeded()) {
                        auto string_ref = llvm::StringRef(string_result);
                        param_names_vec.push_back(string_ref);
                        return mlir::success();
                    } else {
                        return mlir::failure();
                    }
                });
            if (parse_result.succeeded()) {
                auto param_names_array_ref =
                    llvm::ArrayRef<llvm::StringRef>(param_names_vec);
                parameterNames = builder.getStrArrayAttr(param_names_array_ref);
            } else {
                return mlir::failure();
            }
        } else {
            functionLike = builder.getBoolAttr(false);
            // TODO(bpp): Check that no parameters are provided?
        }

        return mlir::success();
    }

    void printMacroParametersCommon(
        mlir::OpAsmPrinter &printer,
        mlir::StringAttr macroName,
        mlir::BoolAttr functionLike,
        mlir::ArrayAttr parameterNames) {
        llvm::raw_ostream &os = printer.getStream();

        os << '"' << macroName.getValue();
        if (functionLike.getValue()) {
            int i = 0;
            os << '(';
            auto range = parameterNames.getAsValueRange<mlir::StringAttr>();
            for (const auto &&param_name : range) {
                if (i > 0) {
                    os << ", ";
                }
                os << param_name;
                i++;
            }
            os << ')';
        }
        os << '"';
    }

    void printMacroParameters(
        mlir::OpAsmPrinter &printer,
        MacroExpansionExpr op,
        mlir::StringAttr macroName,
        mlir::BoolAttr functionLike,
        mlir::ArrayAttr parameterNames) {
        printMacroParametersCommon(printer, macroName,
                                   functionLike, parameterNames);
    }



    void printMacroParameters(
        mlir::OpAsmPrinter &printer,
        MacroExpansionStmt op,
        mlir::StringAttr macroName,
        mlir::BoolAttr functionLike,
        mlir::ArrayAttr parameterNames) {
        printMacroParametersCommon(printer, macroName,
                                   functionLike, parameterNames);
    }

}

#define GET_OP_CLASSES
#include "macroni/Dialect/Macroni/Macroni.cpp.inc"