// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Macroni/MacroniOps.hpp"
#include "macroni/Dialect/Macroni/MacroniDialect.hpp"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::macroni {

void MacroExpansion::build(mlir::OpBuilder &odsBuilder,
                           mlir::OperationState &odsState,
                           mlir::StringAttr macroName,
                           mlir::ArrayAttr parameterNames,
                           mlir::BoolAttr functionLike, mlir::Type rty,
                           std::unique_ptr<mlir::Region> &&region) {
  mlir::OpBuilder::InsertionGuard guard(odsBuilder);

  odsState.addAttribute("macroName", macroName);
  odsState.addAttribute("parameterNames", parameterNames);
  odsState.addAttribute("functionLike", functionLike);
  odsState.addRegion(std::move(region));
  if (rty) {
    odsState.addTypes(rty);
  }
}

void MacroParameter::build(mlir::OpBuilder &odsBuilder,
                           mlir::OperationState &odsState,
                           mlir::StringAttr parameterName, mlir::Type rty,
                           std::unique_ptr<mlir::Region> &&region) {
  mlir::OpBuilder::InsertionGuard guard(odsBuilder);

  odsState.addAttribute("parameterName", parameterName);
  odsState.addRegion(std::move(region));
  if (rty) {
    odsState.addTypes(rty);
  }
}

mlir::ParseResult parseMacroParameters(mlir::OpAsmParser &parser,
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

  // NOTE(bpp): Use optional LParen or mandatory LParen here? Not sure what the
  // difference is if we check for success either way
  parse_result = parser.parseOptionalLParen();
  if (parse_result.succeeded()) {
    functionLike = builder.getBoolAttr(true);
    llvm::SmallVector<llvm::StringRef, 4> param_names_vec;
    parse_result = parser.parseCommaSeparatedList([&]() {
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

void printMacroParameters(mlir::OpAsmPrinter &printer, MacroExpansion op,
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
} // namespace macroni::macroni

#define GET_OP_CLASSES
#include "macroni/Dialect/Macroni/Macroni.cpp.inc"