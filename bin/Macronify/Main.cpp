// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.


#include <clang/AST/ASTContext.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

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

    // NOTE(bpp): I think it would be a good idea to also transform
    // substitutions of get_user's parameters into special operations as well,
    // to even more information about the macro. This would let us match against
    // all the various definitions of get_user, and all its substitutions of all
    // its parameters.

    struct macro_expr_to_get_user
        : mlir::OpConversionPattern< macroni::MacroExpansionExpr > {
        using parent_t = mlir::OpConversionPattern<macroni::MacroExpansionExpr>;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
            macroni::MacroExpansionExpr op,
            macroni::MacroExpansionExpr::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
            if (op.getMacroName() != "get_user") {
                return mlir::failure();
            }
            if (op.getParameterNames().size() != 2) {
                return mlir::failure();
            }

            std::optional<macroni::MacroParameterExpr> last_x = std::nullopt;
            std::optional<macroni::MacroParameterExpr> last_ptr = std::nullopt;

            op.getExpansion().getDefiningOp()->walk(
                [&](mlir::Operation *op) {
                    if (auto param_op = mlir::dyn_cast<macroni::MacroParameterExpr>(op)) {
                        auto param_name = param_op.getParameterName();
                        if (param_name == "x") {
                            last_x.emplace(param_op);
                        } else if (param_name == "ptr") {
                            last_ptr.emplace(param_op);
                        }
                    }
                }
            );

            auto last_x_clone =
                rewriter.clone(*last_x->getExpansion().getDefiningOp());
            auto last_ptr_clone = 
                rewriter.clone(*last_ptr->getExpansion().getDefiningOp());

            mlir::Type result_type = op.getType();
            // mlir::Value x = rewriter.create<vast::hl::ConstantOp>(op.getLoc(), rewriter.getI32Type(), llvm::APSInt("42"));
            mlir::Value x = last_x_clone->getResult(0);
            // mlir::Value ptr = rewriter.create<vast::hl::ConstantOp>(op.getLoc(), rewriter.getI32Type(), llvm::APSInt("42"));
            mlir::Value ptr = last_ptr_clone->getResult(0);

            // Erase the original expansion
            //////
            rewriter.eraseOp(op.getExpansion().getDefiningOp());
            //////
            rewriter.replaceOpWithNewOp<macroni::GetUser>(op, result_type,
                                                          x, ptr);
            return mlir::success();
        }
    };

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
        using TypeVisitor =
            vast::cg::CodeGenTypeVisitorWithDataLayoutMixin< Derived >;

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
            bool function_like = false;
            std::vector<llvm::StringRef> param_names;
            if (auto sub = pasta::MacroSubstitution::From(*lowest_macro)) {
                if (auto name = sub->NameOrOperator()) {
                    macro_name = name->Data();
                }
                if (auto exp = pasta::MacroExpansion::From(*sub)) {
                    if (auto def = exp->Definition()) {
                        function_like = def->IsFunctionLike();
                        for (auto macro_tok : def->Parameters()) {
                            if (auto bt = macro_tok.BeginToken()) {
                                param_names.push_back(bt->Data());
                            }
                        }
                    }
                }
            }

            auto loc = StmtVisitor::meta_location(stmt);
            vast::Operation *result = nullptr;
            visiting.insert(*lowest_macro);
            if (const auto expr = clang::dyn_cast<clang::Expr>(stmt)) {
                auto expansion = StmtVisitor::visit(expr)->getResult(0);
                if (lowest_macro->Kind() ==
                    pasta::MacroKind::kParameterSubstitution) {
                    result =
                        StmtVisitor::template make<macroni::MacroParameterExpr>(
                            loc,
                            expansion,
                            macro_name
                        );
                } else {
                    result =
                        StmtVisitor::template make<macroni::MacroExpansionExpr>(
                            loc,
                            expansion,
                            macro_name,
                            builder->getStrArrayAttr(
                                llvm::ArrayRef(param_names)),
                            function_like
                        );
                }
            } else {
                auto expansion_builder = StmtVisitor::make_region_builder(stmt);
                if (lowest_macro->Kind() ==
                    pasta::MacroKind::kParameterSubstitution) {
                    result = StmtVisitor::template
                        make < macroni::MacroParameterStmt >(
                            loc,
                            expansion_builder,
                            builder->getStringAttr(llvm::Twine(macro_name))
                        );
                } else {
                    result = StmtVisitor::template
                        make< macroni::MacroExpansionStmt >(
                            loc,
                            expansion_builder,
                            builder->getStringAttr(llvm::Twine(macro_name)),
                            builder->getStrArrayAttr(
                                llvm::ArrayRef(param_names)),
                            builder->getBoolAttr(function_like)
                        );
                }
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

    auto maybe_cwd = (pasta::FileSystem::From(maybe_compiler.Value())
                      ->CurrentWorkingDirectory());
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

        // Register the MLIR dialects we will be converting to
        registry.insert<
            vast::hl::HighLevelDialect,
            macroni::macroni::MacroniDialect
        >();
        mctx.emplace(registry);
        macroni::MetaGenerator meta(*ast, &*mctx);
        vast::cg::CodeGenBase<macroni::Visitor> codegen(&*mctx, meta);
        builder.emplace(&*mctx);

        // Generate the MLIR code and freeze the result
        codegen.append_to_module(ast->UnderlyingAST().getTranslationUnitDecl());
        mlir::OwningOpRef<mlir::ModuleOp> mod = codegen.freeze();

        // TODO(bpappas): Add a command-line argument to convert special macros
        // into special operations

        // Register conversions
        mlir::ConversionTarget trg(*mctx);
        // TODO(bpappas): Apparently MLIR will only transform illegal
        // operations? I will need to dynamically make MacroExpansionExprs legal
        // only if they are not invocations of get_user. I will probably have to
        // do something similar for get_user's arguments.
        trg.addIllegalOp<macroni::macroni::MacroExpansionExpr>();
        trg.markUnknownOpDynamicallyLegal([](auto) { return true; });
        mlir::RewritePatternSet patterns(&*mctx);
        patterns.add<macroni::macro_expr_to_get_user>(patterns.getContext());

        // Apply the conversions. Cast the result to void to ignore no_discard
        // errors
        (void) mlir::applyPartialConversion(mod.get().getOperation(),
                                            trg,
                                            std::move(patterns));

        // Print the result
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(false, false);
        mod->print(llvm::outs(), flags);

        return EXIT_SUCCESS;
    }
    return EXIT_SUCCESS;
}
