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

    // NOTE(bpp): It may be a good idea to also transform substitutions of
    // get_user()'s parameters into special operations as well, to get even
    // more information about the macro. This would let us match against all the
    // various definitions of get_user, and all its substitutions of all its
    // parameters.

    struct macro_expansion_to_get_user
        : mlir::OpConversionPattern< macroni::MacroExpansion > {
        using parent_t = mlir::OpConversionPattern<macroni::MacroExpansion>;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
            macroni::MacroExpansion op,
            macroni::MacroExpansion::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
            // TODO(bpp): Add more checks here?
            if (op.getMacroName() != "get_user") {
                return mlir::failure();
            }
            if (op.getResultTypes().empty()) {
                return mlir::failure();
            }
            if (op.getParameterNames().size() != 2) {
                return mlir::failure();
            }

            // Find the last-parsed expansions of the x and ptr parameters.
            // NOTE(bpp): This assumes that all expansions of these
            // parameters will parse the same.

            std::optional<macroni::MacroParameter> last_x = std::nullopt;
            std::optional<macroni::MacroParameter> last_ptr = std::nullopt;

            op.getExpansion().walk(
                [&](mlir::Operation *op) {
                    if (auto param_op = mlir::dyn_cast<macroni::MacroParameter>(op)) {
                        auto param_name = param_op.getParameterName();
                        if (param_name == "x") {
                            last_x.emplace(param_op);
                        } else if (param_name == "ptr") {
                            last_ptr.emplace(param_op);
                        }
                    }
                }
            );

            // Check that we actually found a parameter substitution of the x
            // and ptr parameters
            if (!(last_x && last_ptr)) {
                return mlir::failure();
            }

            // Clone the arguments passed to this invocation of get_user() and
            // lift them out of the expansion.
            auto last_x_clone = rewriter.clone(*last_x->getOperation());
            auto last_ptr_clone = rewriter.clone(*last_ptr->getOperation());

            // Create the replacement operation.
            mlir::Type result_type = op.getType(0);
            mlir::Value x = last_x_clone->getResult(0);
            mlir::Value ptr = last_ptr_clone->getResult(0);
            rewriter.replaceOpWithNewOp<macroni::GetUser>(op, result_type,
                                                          x, ptr);

            return mlir::success();
        }
    };

    struct MacroniMetaGenerator {
        MacroniMetaGenerator(const pasta::AST &ast, mlir::MLIRContext *mctx)
            : ast(ast), mctx(mctx) {}

        vast::cg::DefaultMeta get(const clang::FullSourceLoc &loc) const {
            if (!loc.isValid()) {
                return { mlir::FileLineColLoc::get(mctx, "<invalid>", 0, 0) };
            }
            auto file_entry = loc.getFileEntry();
            auto file = file_entry ? file_entry->getName() : "unknown";
            auto line = loc.getLineNumber();
            auto col = loc.getColumnNumber();
            return { mlir::FileLineColLoc::get(mctx, file, line, col) };
        }

        vast::cg::DefaultMeta get(const clang::SourceLocation &loc) const {
            clang::SourceManager &sm = ast.UnderlyingAST().getSourceManager();
            return get(clang::FullSourceLoc(loc, sm));
        }

        vast::cg::DefaultMeta get(const clang::Decl *decl) const {
            return get(decl->getLocation());
        }

        vast::cg::DefaultMeta get(const clang::Stmt *stmt) const {
            // TODO: use SourceRange
            return get(stmt->getBeginLoc());
        }

        vast::cg::DefaultMeta get(const clang::Expr *expr) const {
            // TODO: use SourceRange
            return get(expr->getExprLoc());
        }

        vast::cg::DefaultMeta get(const clang::TypeLoc &loc) const {
            // TODO: use SourceRange
            return get(loc.getBeginLoc());
        }

        vast::cg::DefaultMeta get(const clang::Type *type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        vast::cg::DefaultMeta get(clang::QualType type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        vast::cg::DefaultMeta get(pasta::MacroSubstitution &sub) const {
            // TODO(bpp): Define this to something that makes sense. Right now
            // this just returns an invalid source location.
            return get(clang::SourceLocation());
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

        std::set<pasta::MacroSubstitution> visited;

        mlir::Operation *Visit(const clang::Stmt *stmt) {
            if (clang::isa<clang::ImplicitValueInitExpr,
                clang::ImplicitCastExpr>(stmt)) {
                // Don't visit implicit expressions
                return StmtVisitor::Visit(stmt);
            }

            // Find the lowest macro that covers this statement, if any
            const auto pasta_stmt = ast->Adopt(stmt);
            std::optional<pasta::MacroSubstitution> lowest_sub = std::nullopt;
            auto subs = pasta_stmt.AlignedSubstitutions();
            std::reverse(subs.begin(), subs.end());
            for (auto sub : subs) {
                // Don't visit macros more than once
                if (visited.contains(sub)) {
                    continue;
                }

                // Only visit pre-expanded forms of function-like expansions.
                if (auto exp = pasta::MacroExpansion::From(sub)) {
                    bool is_pre_expansion = (exp->Arguments().empty() ||
                                             exp->IsArgumentPreExpansion());
                    if (!is_pre_expansion) {
                        continue;
                    }
                }

                // Mark this substitution as visited so we don't visit it again.
                visited.insert(sub);
                lowest_sub = sub;
                break;
            }

            if (!lowest_sub) {
                // If no substitution covers this statement, visit it normally.
                return StmtVisitor::Visit(stmt);
            }

            // Get the substitution's location, name, parameter names, and
            // whether it is function-like
            mlir::Location loc = StmtVisitor::meta_location(*lowest_sub);
            std::string_view macro_name = "<a nameless macro>";
            bool function_like = false;
            std::vector<llvm::StringRef> param_names;
            if (auto sub = pasta::MacroSubstitution::From(*lowest_sub)) {
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

            // We call `make_maybe_value_yield_region` here because a macro may
            // not expand to an expression
            auto [region, return_type] =
                StmtVisitor::make_maybe_value_yield_region(stmt);

            // Check if the macro is an expansion or a parameter, and return the
            // appropriate operation
            if (lowest_sub->Kind() == pasta::MacroKind::kExpansion) {
                return StmtVisitor::template make<macroni::MacroExpansion>(
                    loc,
                    builder->getStringAttr(llvm::Twine(macro_name)),
                    builder->getStrArrayAttr(llvm::ArrayRef(param_names)),
                    builder->getBoolAttr(function_like),
                    return_type,
                    std::move(region)
                );
            } else {
                return StmtVisitor::template make<macroni::MacroParameter>(
                    loc,
                    builder->getStringAttr(llvm::Twine(macro_name)),
                    return_type,
                    std::move(region)
                );
            }
        }
    };

    template< typename Derived >
    using VisitorConfig = vast::cg::CodeGenFallBackVisitorMixin< Derived,
        CodeGenVisitorMixin,
        vast::cg::DefaultFallBackVisitorMixin
    >;

    using Visitor =
        vast::cg::CodeGenVisitor<VisitorConfig, MacroniMetaGenerator>;

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
        macroni::MacroniMetaGenerator meta(*ast, &*mctx);
        vast::cg::CodeGenBase<macroni::Visitor> codegen(&*mctx, meta);
        builder.emplace(&*mctx);

        // Generate the MLIR code and freeze the result
        codegen.append_to_module(ast->UnderlyingAST().getTranslationUnitDecl());
        mlir::OwningOpRef<mlir::ModuleOp> mod = codegen.freeze();

        // TODO(bpp): Add a command-line argument to convert special macros into
        // special operations

        // Register conversions
        mlir::ConversionTarget trg(*mctx);
        // TODO(bpp): Apparently MLIR will only transform illegal operations? I
        // will need to dynamically make MacroExpansions legal only if they are
        // not invocations of get_user. I will probably have to do something
        // similar for get_user's arguments.
        trg.addIllegalOp<macroni::macroni::MacroExpansion>();
        trg.markUnknownOpDynamicallyLegal([](auto) { return true; });
        mlir::RewritePatternSet patterns(&*mctx);
        patterns.add<macroni::macro_expansion_to_get_user>(patterns.getContext());
        mlir::Operation *mod_op = mod.get().getOperation();
        // Apply the conversions. Cast the result to void to ignore no_discard
        // errors
        (void) mlir::applyPartialConversion(mod_op, trg, std::move(patterns));

        // Print the result
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(false, true);
        mod->print(llvm::outs(), flags);

        return EXIT_SUCCESS;
    }
    return EXIT_SUCCESS;
}
