#pragma once

#include <mlir/IR/MLIRContext.h>
#include <pasta/AST/AST.h>
#include <vast/CodeGen/CodeGenContext.hpp>

namespace macroni {
struct MacroniCodeGenContext : public vast::cg::codegen_context {
  pasta::AST &pasta_ast;

  MacroniCodeGenContext(mlir::MLIRContext &mctx, clang::ASTContext &actx,
                        pasta::AST &pasta_ast)
      : codegen_context(mctx, actx,
                        vast::cg::detail::create_module(
                            mctx, actx,
                            // NOTE(bpp): For now we only support C
                            vast::core::SourceLanguage::C)),
        pasta_ast(pasta_ast) {}
};
} // namespace macroni
