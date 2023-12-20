#pragma once

#include <macroni/Common/GenerateMacroniModule.hpp>
#include <macroni/Translation/MacroniCodeGenContext.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <pasta/AST/AST.h>

namespace macroni {

template <typename MacroniCodeGenInstanceType>
mlir::OwningOpRef<mlir::ModuleOp>
generate_macroni_module(pasta::AST &pasta_ast, mlir::MLIRContext &mctx) {
  auto &actx = pasta_ast.UnderlyingAST();
  auto cgctx = MacroniCodeGenContext(mctx, actx, pasta_ast);
  auto meta = MacroniMetaGenerator(&actx, &mctx);
  auto codegen_instance = MacroniCodeGenInstanceType(cgctx, meta);

  codegen_instance.emit_data_layout();
  codegen_instance.Visit(actx.getTranslationUnitDecl());
  codegen_instance.verify_module();

  return std::move(cgctx.mod);
}

} // namespace macroni
