// Copyright (c) 2022-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.


#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>

VAST_UNRELAX_WARNINGS

#include <vast/Util/Common.hpp>
#include <vast/Translation/CodeGenContext.hpp>
#include <vast/Translation/CodeGenVisitor.hpp>
#include <vast/Dialect/Dialects.hpp>
#include <vast/Conversion/Passes.hpp>
#include <vast/Translation/CodeGen.hpp>
#include <vast/Translation/Register.hpp>
#include <vast/Translation/CodeGenDriver.hpp>
#include <vast/Translation/CodeGenTypeDriver.hpp>
#include <mlir/Tools/mlir-translate/Translation.h>

#include <pasta/AST/AST.h>
#include <pasta/AST/Decl.h>
#include <pasta/Compile/Command.h>
#include <pasta/Compile/Compiler.h>
#include <pasta/Compile/Job.h>
#include <pasta/Util/ArgumentVector.h>
#include <pasta/Util/FileSystem.h>
#include <pasta/Util/Init.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <optional>

// TODO(bpp): Instead of using a global variable for the PASTA AST, find out how
// to pass it to the CodeGen
std::optional<pasta::AST> ast;

namespace macroni {

    struct MetaGenerator {
        MetaGenerator(const pasta::AST &ast, mlir::MLIRContext *mctx)
            : ast(ast), mctx(mctx) {}

        vast::cg::DefaultMeta get(const clang::FullSourceLoc &loc) const {
            return { mlir::FileLineColLoc::get(mctx, "<input>", 0, 0) };
        }

        vast::cg::DefaultMeta get(const clang::SourceLocation &loc) const {
            return get(clang::FullSourceLoc());
        }

        vast::cg::DefaultMeta get(const clang::Decl *decl) const {
            return get(clang::SourceLocation());
        }

        vast::cg::DefaultMeta get(const clang::Stmt *stmt) const {
            // TODO: use SourceRange
            return get(clang::SourceLocation());
        }

        vast::cg::DefaultMeta get(const clang::Expr *expr) const {
            // TODO: use SourceRange
            return get(clang::SourceLocation());
        }

        vast::cg::DefaultMeta get(const clang::TypeLoc &loc) const {
            // TODO: use SourceRange
            return get(clang::SourceLocation());
        }

        vast::cg::DefaultMeta get(const clang::Type *type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        vast::cg::DefaultMeta get(clang::QualType type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        const pasta::AST &ast;
        mlir::MLIRContext *mctx;
    };

    template< typename Derived >
    struct CodeGenVisitorMixin
        : vast::cg::CodeGenDeclVisitorMixin< Derived >
        , vast::cg::CodeGenStmtVisitorMixin< Derived >
        , vast::cg::CodeGenTypeVisitorWithDataLayoutMixin< Derived >
    {
        using DeclVisitor = vast::cg::CodeGenDeclVisitorMixin< Derived >;
        using StmtVisitor = vast::cg::CodeGenStmtVisitorMixin< Derived >;
        using TypeVisitor = vast::cg::CodeGenTypeVisitorWithDataLayoutMixin< Derived >;

        using DeclVisitor::Visit;
        using TypeVisitor::Visit;

        mlir::Operation *Visit(const clang::Stmt *stmt) {
            auto pasta_stmt = ast->Adopt(stmt);
            if (pasta_stmt.LowestCoveringMacro(pasta::MacroKind::kExpansion)) {
                std::cerr << "Test\n";
            }
            return StmtVisitor::Visit(stmt);
        }
    };

    template< typename Derived >
    using VisitorConfig = vast::cg::CodeGenFallBackVisitorMixin< Derived,
        CodeGenVisitorMixin,
        vast::cg::DefaultFallBackVisitorMixin
    >;

    using Visitor = vast::cg::CodeGenVisitor<VisitorConfig, MetaGenerator>;

} // namespace macroni

static llvm::cl::list< std::string > compiler_args(
    "ccopts", llvm::cl::ZeroOrMore, llvm::cl::desc("Specify compiler options")
);

int main(int argc, char **argv) {

    pasta::InitPasta initializer;
    pasta::FileManager fm(pasta::FileSystem::CreateNative());
    auto maybe_compiler =
        pasta::Compiler::CreateHostCompiler(fm, pasta::TargetLanguage::kCXX);
    if (!maybe_compiler.Succeeded()) {
        std::cerr << maybe_compiler.TakeError() << std::endl;
        return EXIT_FAILURE;
    }

    auto maybe_cwd = pasta::FileSystem::From(maybe_compiler.Value())->CurrentWorkingDirectory();
    if (!maybe_cwd.Succeeded()) {
        std::cerr << maybe_compiler.TakeError() << std::endl;
        return EXIT_FAILURE;
    }

    const pasta::ArgumentVector args(argc - 1, &argv[1]);
    auto maybe_command = pasta::CompileCommand::CreateFromArguments(
        args, maybe_cwd.TakeValue());
    if (!maybe_command.Succeeded()) {
        std::cerr << maybe_command.TakeError() << std::endl;
        return EXIT_FAILURE;
    }

    const auto command = maybe_command.TakeValue();
    auto maybe_jobs = maybe_compiler->CreateJobsForCommand(command);
    if (!maybe_jobs.Succeeded()) {
        std::cerr << maybe_jobs.TakeError() << std::endl;
        return EXIT_FAILURE;
    }

    for (const auto &job : maybe_jobs.TakeValue()) {
        auto maybe_ast = job.Run();
        if (!maybe_ast.Succeeded()) {
            std::cerr << maybe_ast.TakeError() << std::endl;
            return EXIT_FAILURE;
        }
        else {
            ast.emplace(maybe_ast.TakeValue());

            mlir::DialectRegistry registry;
            registry.insert<vast::hl::HighLevelDialect>();
            mlir::MLIRContext mctx(registry);
            macroni::MetaGenerator meta(*ast, &mctx);
            vast::cg::CodeGenBase<macroni::Visitor> codegen(&mctx, meta);

            codegen.append_to_module(ast->UnderlyingAST().getTranslationUnitDecl());
            auto mod = codegen.freeze();
            mod->print(llvm::outs());

            return EXIT_SUCCESS;
        }
    }
    return EXIT_SUCCESS;
}
