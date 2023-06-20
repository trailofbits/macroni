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
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Support/TypeID.h>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>

#include <pasta/AST/AST.h>
#include <pasta/AST/Decl.h>
#include <pasta/Compile/Command.h>
#include <pasta/Compile/Compiler.h>
#include <pasta/Compile/Job.h>
#include <pasta/Util/ArgumentVector.h>
#include <pasta/Util/FileSystem.h>
#include <pasta/Util/Init.h>

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

#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <optional>
#include <algorithm>
#include <set>
#include <map>

// TODO(bpp): Instead of using a global variable for the PASTA AST and MLIR
// context, find out how to pass these to a CodeGen object.
std::optional<pasta::AST> ast = std::nullopt;
std::optional<mlir::MLIRContext> mctx = std::nullopt;
std::optional<mlir::Builder> builder = std::nullopt;

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

    // Adds an MLIR StrAttr with the given name and value to an MLIR operation.
    void AddStrAttr(mlir::Operation *op, std::string_view name, std::string_view value) {
        auto attr_name = llvm::StringRef(name);
        auto twine = llvm::Twine(value);
        auto attr = mlir::StringAttr::get(&*mctx, twine);
        op->setAttr(attr_name, attr);
    }

    // Adds an MLIR U64Attr with the given name and value to an MLIR operation.
    void AddU64Attr(mlir::Operation *op, std::string_view name, uint64_t value) {
        auto attr_name = llvm::StringRef(name);
        auto aps_int = mlir::APSInt(llvm::APInt(64, value, false), true);
        auto attr = mlir::IntegerAttr::get(&*mctx, aps_int);
        op->setAttr(attr_name, attr);
    }

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

        using HLScope = vast::cg::HighLevelScope;

        std::set<pasta::Macro> visiting;

        mlir::Operation *Visit(const clang::Stmt *stmt) {
            if (clang::isa<clang::ImplicitValueInitExpr,
                clang::ImplicitCastExpr>(stmt)) {
                // Don't visit implicit expressions
                return StmtVisitor::Visit(stmt);
            }

            // Find the lowest macro that covers this statement, if any
            const auto pasta_stmt = ast->Adopt(stmt);
            std::optional<pasta::Macro> lowest_macro = std::nullopt;
            auto macros = pasta_stmt.CoveringMacros();
            std::reverse(macros.begin(), macros.end());
            for (auto macro : macros) {

                // Don't visit macros more than once
                if (visiting.contains(macro)) {
                    continue;
                }

                // Only visit pre-expanded forms of function-like macro
                // expansions.
                if (auto exp = pasta::MacroExpansion::From(macro)) {
                    bool is_pre_expansion = (exp->Arguments().empty() ||
                                             exp->IsArgumentPreExpansion());
                    if (!is_pre_expansion) {
                        continue;
                    }
                }

                // Don't visit macro arguments, only their substitutions
                if (auto arg = pasta::MacroArgument::From(macro)) {
                    continue;
                }

                lowest_macro = macro;
                break;
            }

            if (!lowest_macro.has_value()) {
                // If no macro covers this statement, visit it normally.
                return StmtVisitor::Visit(stmt);
            }

            // First get the macro's name
            std::string_view macro_name = "<a nameless macro>";
            std::string_view macro_kind = lowest_macro->KindName();
            uintptr_t macro_id =
                reinterpret_cast<uintptr_t>(lowest_macro->RawMacro());
            auto macro_id_ap_int = llvm::APInt(64, macro_id, false);
            bool function_like = false;
            std::vector<llvm::StringRef> parameter_names;
            if (auto sub = pasta::MacroSubstitution::From(*lowest_macro)) {
                if (auto name = sub->NameOrOperator()) {
                    macro_name = name->Data();
                }
                if (auto exp = pasta::MacroExpansion::From(*sub)) {
                    if (auto def = exp->Definition()) {
                        function_like = def->IsFunctionLike();
                        for (auto macro_tok : def->Parameters()) {
                            if (auto bt = macro_tok.BeginToken()) {
                                parameter_names.push_back(bt->Data());
                            }
                        }
                    }
                }
            }

            auto loc = StmtVisitor::meta_location(stmt);
            vast::Operation *result = nullptr;
            visiting.insert(*lowest_macro);
            if (const auto expr = clang::dyn_cast<clang::Expr>(stmt)) {
                // If this macro is an expression, replace it with a
                // MacroExpansionExpr.
                auto expansion = StmtVisitor::visit(expr)->getResult(0);
                result = StmtVisitor::template make<macroni::MacroExpansionExpr>(
                    loc,
                    expansion,
                    macro_id,
                    macro_name,
                    macro_kind,
                    builder->getStrArrayAttr(llvm::ArrayRef(parameter_names)),
                    function_like
                );
            } else {
                // Otherwise, replace it with a MacroExpansionStmt.
                auto expansion_builder = StmtVisitor::make_region_builder(stmt);
                result = StmtVisitor::template make< macroni::MacroExpansionStmt >(
                    loc,
                    expansion_builder,
                    builder->getIntegerAttr(builder->getI64Type(),
                                            macro_id_ap_int),
                    builder->getStringAttr(llvm::Twine(macro_name)),
                    builder->getStringAttr(llvm::Twine(macro_kind)),
                    builder->getStrArrayAttr(llvm::ArrayRef(parameter_names)),
                    builder->getBoolAttr(function_like)
                );
            }
            visiting.erase(*lowest_macro);

            return result;
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

        ast.emplace(maybe_ast.TakeValue());

        mlir::DialectRegistry registry;
        registry.insert<
            vast::hl::HighLevelDialect,
            macroni::macroni::MacroniDialect>();
        mctx.emplace(registry);
        macroni::MetaGenerator meta(*ast, &*mctx);
        vast::cg::CodeGenBase<macroni::Visitor> codegen(&*mctx, meta);
        builder.emplace(&*mctx);

        codegen.append_to_module(ast->UnderlyingAST().getTranslationUnitDecl());
        auto mod = codegen.freeze();
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(false, false);
        mod->print(llvm::outs(), flags);

        return EXIT_SUCCESS;
    }
    return EXIT_SUCCESS;
}
