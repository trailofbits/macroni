#pragma once

#include <pasta/AST/AST.h>
#include <vast/CodeGen/CodeGenContext.hpp>

namespace macroni
{
    struct MacroniCodeGenContext : vast::cg::CodeGenContext {
        pasta::AST &pasta_ast;

        MacroniCodeGenContext(mlir::MLIRContext &mctx,
                              clang::ASTContext &actx,
                              pasta::AST &pasta_ast)
            : CodeGenContext(mctx, actx,
                             vast::cg::detail::create_module(mctx, actx)),
                             pasta_ast(pasta_ast)
        {}

    };
} // namespace macroni
