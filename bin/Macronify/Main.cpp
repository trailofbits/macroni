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
#include <macroni/Dialect/Kernel/KernelDialect.hpp>
#include <macroni/Dialect/Kernel/KernelOps.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <optional>
#include <algorithm>
#include <set>
#include <map>

using Op = mlir::Operation;

static const constexpr char get_user[] = "get_user";
static const constexpr char offsetof[] = "offsetof";
static const constexpr char container_of[] = "container_of";
static const constexpr char rcu_read_lock[] = "rcu_read_lock";
static const constexpr char rcu_read_unlock[] = "rcu_read_unlock";

// TODO(bpp): Instead of using a global variable for the PASTA AST and MLIR
// context, find out how to pass these to a CodeGen object.
std::optional<pasta::AST> ast = std::nullopt;
std::optional<mlir::MLIRContext> mctx = std::nullopt;
std::optional<mlir::Builder> builder = std::nullopt;

// TODO(bpp): Find a way to pass information along from the dynamic legality
// callbacks to the conversion methods without using global variables
std::map<Op *, Op *> get_user_to_x;
std::map<Op *, Op *> get_user_to_ptr;

std::map<Op *, mlir::TypeAttr> offsetof_to_type;
std::map<Op *, mlir::StringAttr> offsetof_to_member;

std::map<Op *, Op *> container_of_to_ptr;
std::map<Op *, mlir::TypeAttr> container_of_to_type;
std::map<Op *, mlir::StringAttr> container_of_to_member;

std::map<Op *, Op *> list_for_each_to_pos;
std::map<Op *, Op *> list_for_each_to_head;

std::map<Op *, Op *> unlock_to_lock;

static inline llvm::APInt get_lock_level(Op *op) {
    return op->getAttrOfType<mlir::IntegerAttr>("lock_level").getValue();
}

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
            macroni::MacroExpansion exp,
            macroni::MacroExpansion::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {

            if (exp.getMacroName() != get_user) {
                return mlir::failure();
            }

            Op *op = exp.getOperation(),
                *x = get_user_to_x.at(op),
                *ptr = get_user_to_ptr.at(op),
                *x_clone = rewriter.clone(*x),
                *ptr_clone = rewriter.clone(*ptr);
            mlir::Type result_type = exp.getType(0);
            mlir::Value x_val = x_clone->getResult(0),
                ptr_val = ptr_clone->getResult(0);

            using GE = ::macroni::kernel::GetUser;
            rewriter.replaceOpWithNewOp<GE>(op, result_type, x_val, ptr_val);

            return mlir::success();
        }
    };

    struct macro_expansion_to_offsetof
        : mlir::OpConversionPattern< macroni::MacroExpansion > {
        using parent_t = mlir::OpConversionPattern<macroni::MacroExpansion>;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
            macroni::MacroExpansion exp,
            macroni::MacroExpansion::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {

            if (exp.getMacroName() != offsetof) {
                return mlir::failure();
            }

            Op *op = exp.getOperation();
            mlir::TypeAttr type = offsetof_to_type.at(op);
            mlir::StringAttr member = offsetof_to_member.at(op);
            mlir::Type result_type = exp.getType(0);

            using OO = ::macroni::kernel::OffsetOf;
            rewriter.replaceOpWithNewOp<OO>(op, result_type, type, member);

            return mlir::success();
        }
    };

    struct macro_expansion_to_container_of
        : mlir::OpConversionPattern< macroni::MacroExpansion > {
        using parent_t = mlir::OpConversionPattern<macroni::MacroExpansion>;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
            macroni::MacroExpansion exp,
            macroni::MacroExpansion::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
            if (exp.getMacroName() != container_of) {
                return mlir::failure();
            }

            Op *op = exp.getOperation(),
                *ptr = container_of_to_ptr.at(op),
                *ptr_clone = rewriter.clone(*ptr);
            mlir::TypeAttr type = container_of_to_type.at(op);
            mlir::StringAttr member = container_of_to_member.at(op);
            mlir::Type res_ty = exp.getType(0);
            mlir::Value ptr_val = ptr_clone->getResult(0);

            using CO = ::macroni::kernel::ContainerOf;
            rewriter.replaceOpWithNewOp<CO>(op, res_ty, ptr_val, type, member);

            return mlir::success();
        }
    };

    struct for_to_list_for_each
        : mlir::OpConversionPattern< vast::hl::ForOp > {
        using parent_t = mlir::OpConversionPattern<vast::hl::ForOp>;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
            vast::hl::ForOp for_op,
            vast::hl::ForOp::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
            Op *op = for_op.getOperation(),
                *pos = list_for_each_to_pos.at(op),
                *head = list_for_each_to_head.at(op),
                *pos_clone = rewriter.clone(*pos),
                *head_clone = rewriter.clone(*head);

            mlir::Value pos_val = pos_clone->getResult(0),
                head_val = head_clone->getResult(0);

            auto reg = std::make_unique<mlir::Region>();

            reg->takeBody(for_op.getBodyRegion());

            using LFE = ::macroni::kernel::ListForEach;
            rewriter.replaceOpWithNewOp<LFE>(for_op, pos_val, head_val,
                                             std::move(reg));

            return mlir::success();
        }
    };

    struct rcu_read_unlock_to_rcu_critical_section
        : mlir::OpConversionPattern< vast::hl::CallOp > {
        using parent_t = mlir::OpConversionPattern<vast::hl::CallOp>;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
            vast::hl::CallOp rcu_read_unlock_call,
            vast::hl::CallOp::Adaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {

            Op *unlock_op = rcu_read_unlock_call.getOperation(),
                *lock_op = unlock_to_lock.at(unlock_op);

            using CS = kernel::RCUCriticalSection;
            rewriter.setInsertionPointAfter(lock_op);
            CS cs = rewriter.replaceOpWithNewOp<CS>(lock_op);
            mlir::Block *cs_block = rewriter.createBlock(&cs.getBodyRegion());

            Op *op = cs->getNextNode();
            while (op != unlock_op) {
                Op *temp = op->getNextNode();
                op->moveBefore(cs_block, cs_block->end());
                op = temp;
            }
            rewriter.eraseOp(unlock_op);

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

        int64_t lock_level = 0;

        Op *Visit(const clang::Stmt *stmt) {
            if (clang::isa<clang::ImplicitValueInitExpr,
                clang::ImplicitCastExpr>(stmt)) {
                // Don't visit implicit expressions
                return VisitNonMacro(stmt);
            }

            // Find the lowest macro that covers this statement, if any
            const auto pasta_stmt = ast->Adopt(stmt);
            std::optional<pasta::MacroSubstitution> lowest_sub = std::nullopt;
            auto subs = pasta_stmt.AlignedSubstitutions();
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
                return VisitNonMacro(stmt);
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

        Op *VisitNonMacro(const clang::Stmt *stmt) {
            auto op = StmtVisitor::Visit(stmt);
            if (vast::hl::CallOp call_op =
                mlir::dyn_cast<vast::hl::CallOp>(op)) {
                auto name = call_op.getCalleeAttr().getValue();
                if ("rcu_read_lock" == name) {
                    call_op.getOperation()->setAttr(
                        "lock_level",
                        builder->getI64IntegerAttr(lock_level));
                    lock_level++;
                } else if ("rcu_read_unlock" == name) {
                    lock_level--;
                    call_op.getOperation()->setAttr(
                        "lock_level",
                        builder->getI64IntegerAttr(lock_level));
                }
            }
            return op;
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
            macroni::macroni::MacroniDialect,
            macroni::kernel::KernelDialect
        >();
        mctx.emplace(registry);
        macroni::MacroniMetaGenerator meta(*ast, &*mctx);
        vast::cg::CodeGenBase<macroni::Visitor> codegen(&*mctx, meta);
        builder.emplace(&*mctx);

        codegen.append_to_module(ast->UnderlyingAST().getTranslationUnitDecl());
        mlir::OwningOpRef<mlir::ModuleOp> mod = codegen.freeze();

        // TODO(bpp): Add a command-line argument to convert special macros into
        // special operations

        // Register conversions

        // Mark expansions of get_user(), offsetof() and container_of() as
        // illegal
        mlir::ConversionTarget trg(*mctx);
        trg.addDynamicallyLegalOp<macroni::macroni::MacroExpansion>(
            [](Op *op) {
                using ME = macroni::macroni::MacroExpansion;
                ME exp = mlir::dyn_cast<ME>(op);
                bool has_results = !exp.getResultTypes().empty();
                llvm::StringRef macro_name = exp.getMacroName();
                size_t num_params = exp.getParameterNames().size();
                bool is_get_user = (has_results &&
                                    num_params == 2 &&
                                    macro_name == get_user),
                    is_offsetof = (has_results &&
                                   num_params == 2 &&
                                   macro_name == offsetof),
                    is_container_of = (has_results &&
                                       num_params == 3 &&
                                       macro_name == container_of);

                using MP = macroni::macroni::MacroParameter;
                using RMO = vast::hl::RecordMemberOp;
                using PT = vast::hl::PointerType;
                using TA = mlir::TypeAttr;
                if (is_get_user) {
                    exp.getExpansion().walk([&](Op *cur) {
                        if (auto param_op = mlir::dyn_cast<MP>(cur)) {
                            auto param_name = param_op.getParameterName();
                            if (param_name == "x") {
                                get_user_to_x.insert({ op, cur });
                            } else if (param_name == "ptr") {
                                get_user_to_ptr.insert({ op, cur });
                            }
                        }});
                    bool found_op = get_user_to_x.contains(op),
                        found_ptr = get_user_to_ptr.contains(op),
                        is_legal = !(found_op && found_ptr);
                    return is_legal;
                } else if (is_offsetof) {
                    exp.getExpansion().walk([&](Op *cur) {
                        if (auto member_op = mlir::dyn_cast<RMO>(cur)) {
                            mlir::Value record = member_op.getRecord();
                            mlir::Type record_ty = record.getType();
                            if (auto pty = mlir::dyn_cast<PT>(record_ty)) {
                                mlir::Type element_ty = pty.getElementType();
                                auto ty_attr = TA::get(element_ty);
                                offsetof_to_type.insert({ op, ty_attr });
                            } else {
                                auto ty_attr = TA::get(record_ty);
                                offsetof_to_type.insert({ op, ty_attr });
                            }
                            mlir::StringAttr name = member_op.getNameAttr();
                            offsetof_to_member[op] = name;
                        }});
                    bool found_type = offsetof_to_type.contains(op),
                        found_member = offsetof_to_member.contains(op),
                        is_legal = !(found_type && found_member);
                    return is_legal;
                } else if (is_container_of) {
                    exp.getExpansion().walk([&](Op *cur) {
                        if (mlir::isa<MP>(cur)) {
                            container_of_to_ptr.insert({ op, cur });
                        } else if (auto mem_op = mlir::dyn_cast<RMO>(cur)) {
                            mlir::Value record = mem_op.getRecord();
                            mlir::Type record_ty = record.getType();
                            if (auto pty = mlir::dyn_cast<PT>(record_ty)) {
                                mlir::Type elem_ty = pty.getElementType();
                                TA ty_attr = TA::get(elem_ty);
                                container_of_to_type.insert({ op, ty_attr });
                            } else {
                                TA ty_attr = TA::get(record_ty);
                                container_of_to_type.insert({ op, ty_attr });
                            }
                            mlir::StringAttr mem_name = mem_op.getNameAttr();
                            container_of_to_member.insert({ op, mem_name });
                        }});
                    bool found_ptr = container_of_to_ptr.contains(op),
                        found_type = container_of_to_type.contains(op),
                        found_member = container_of_to_member.contains(op),
                        is_legal = !(found_ptr && found_type && found_member);
                    return is_legal;
                } else {
                    return true;
                }
            });
        trg.addDynamicallyLegalOp<vast::hl::ForOp>(
            [](Op *op) {
                auto for_op = mlir::dyn_cast<vast::hl::ForOp>(op);
                using MP = macroni::macroni::MacroParameter;
                for_op.getCondRegion().walk(
                    [&](Op *op2) {
                        if (auto param_op = mlir::dyn_cast<MP>(op2)) {
                            if (param_op.getParameterName() == "pos") {
                                list_for_each_to_pos[op] = op2;
                            } else if (param_op.getParameterName() == "head") {
                                list_for_each_to_head[op] = op2;
                            }
                        }
                    }
                );
                return !(list_for_each_to_pos.contains(op) &&
                         list_for_each_to_head.contains(op));
            });
        trg.addDynamicallyLegalOp<vast::hl::CallOp>(
            [](Op *op) {
                using CO = vast::hl::CallOp;
                CO call_op = mlir::dyn_cast<CO>(op);
                llvm::StringRef name = call_op.getCalleeAttr().getValue();
                if (rcu_read_unlock != name) {
                    return true;
                }
                llvm::APInt unlock_level = get_lock_level(op);
                for (auto cur = op; cur; cur = cur->getPrevNode()) {
                    if (CO call_op = mlir::dyn_cast<CO>(cur)) {
                        name = call_op.getCalleeAttr().getValue();
                        if (rcu_read_lock == name) {
                            llvm::APInt lock_level = get_lock_level(cur);
                            if (unlock_level == lock_level) {
                                unlock_to_lock[op] = cur;
                                break;
                            }
                        }
                    }
                }
                return !unlock_to_lock.contains(op);
            });
        trg.markUnknownOpDynamicallyLegal([](Op *op) { return true;});

        mlir::RewritePatternSet patterns(&*mctx);
        patterns.add<macroni::macro_expansion_to_get_user>(patterns.getContext());
        patterns.add<macroni::macro_expansion_to_offsetof>(patterns.getContext());
        patterns.add<macroni::macro_expansion_to_container_of>(patterns.getContext());
        patterns.add<macroni::for_to_list_for_each>(patterns.getContext());
        patterns.add<macroni::rcu_read_unlock_to_rcu_critical_section>(patterns.getContext());
        Op *mod_op = mod.get().getOperation();
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
