#pragma once

#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <macroni/Translation/MacroniCodeGenContext.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <optional>
#include <pasta/AST/AST.h>
#include <pasta/AST/Macro.h>
#include <pasta/Util/File.h>
#include <set>
#include <vast/CodeGen/CodeGen.hpp>
#include <vast/CodeGen/FallBackVisitor.hpp>
#include <vast/CodeGen/UnreachableVisitor.hpp>
#include <vast/CodeGen/UnsupportedVisitor.hpp>
#include <vector>

namespace macroni {
// Given a set of visited substitutions, returns the lowest substitutions in
// this macros chain of aligned substitution that has not yet been visited, and
// marks it as visited.
std::optional<pasta::MacroSubstitution>
lowest_unvisited_substitution(pasta::Stmt &stmt,
                              std::set<pasta::MacroSubstitution> &visited);

// Given a substitution, returns whether that substitution is an expansion of a
// function-like macro. Conservatively returns false if the substitution lacks
// the necessary information to determine whether it is function-like or not.
bool is_sub_function_like(pasta::MacroSubstitution &sub);

// Given a substitution, returns the names of the names of the substitution's
// macro parameters, if any.
std::vector<llvm::StringRef> get_parameter_names(pasta::MacroSubstitution &sub);

template <typename Derived>
struct MacroniCodeGenVisitorMixin
    : vast::cg::decl_visitor_with_attrs<Derived>,
      vast::cg::default_stmt_visitor<Derived>,
      vast::cg::type_visitor_with_dl<Derived>,
      vast::cg::default_attr_visitor<Derived>,
      vast::cg::visitor_lens<Derived, MacroniCodeGenVisitorMixin> {

  std::set<pasta::MacroSubstitution> visited;

  using decl_visitor = vast::cg::decl_visitor_with_attrs<Derived>;
  using stmt_visitor = vast::cg::default_stmt_visitor<Derived>;
  using type_visitor = vast::cg::type_visitor_with_dl<Derived>;
  using attr_visitor = vast::cg::default_attr_visitor<Derived>;
  using lens = vast::cg::visitor_lens<Derived, MacroniCodeGenVisitorMixin>;

  using attr_visitor::Visit;
  using decl_visitor::Visit;
  using lens::acontext;
  using lens::context;
  using lens::derived;
  using lens::insertion_guard;
  using lens::mcontext;
  using lens::meta_location;
  using lens::mlir_builder;
  using lens::set_insertion_point_to_end;
  using type_visitor::Visit;

  // VAST used to define this function as part of their API, but removed it in
  // favor of `make_stmt_expr_region()`. We redefine it here as a templated
  // function to handle macro substitutions, since macro substitutions may or
  // may not expand to an expression.
  template <typename StmtType>
  std::pair<std::unique_ptr<mlir::Region>, mlir::Type>
  make_maybe_value_yield_region(const StmtType *stmt) {
    auto guard = insertion_guard();
    auto reg = derived().make_stmt_region(stmt);

    auto &block = reg->back();
    auto type = mlir::Type();
    set_insertion_point_to_end(&block);
    if (block.back().getNumResults() > 0) {
      type = block.back().getResult(0).getType();
      mlir_builder().template create<vast::hl::ValueYieldOp>(
          meta_location(stmt), block.back().getResult(0));
    }
    return {std::move(reg), type};
  }

  mlir::Operation *Visit(const clang::Stmt *stmt) {
    auto &codegen_context = static_cast<MacroniCodeGenContext &>(context());
    auto pasta_stmt = codegen_context.pasta_ast.Adopt(stmt);

    if (clang::isa<clang::ImplicitValueInitExpr, clang::ImplicitCastExpr>(
            stmt)) {
      // Don't visit implicit expressions
      auto op = stmt_visitor::Visit(stmt);
      derived().UnalignedStmtVisited(pasta_stmt, op);
      return op;
    }

    // Find the lowest macro that covers this statement, if any
    auto sub = lowest_unvisited_substitution(pasta_stmt, visited);
    if (!sub) {
      // If no substitution covers this statement, visit it normally.
      auto op = stmt_visitor::Visit(stmt);
      derived().UnalignedStmtVisited(pasta_stmt, op);
      return op;
    }

    // Get the substitution's location, name, parameter names, and whether it is
    // function-like.
    //
    // NOTE(bpp): We have to use a dynamic_cast here because
    // vast::cg::codegen_instance expects a vast::cg::meta_generator as its meta
    // generator, but we use static inheritance to pass it our own meta
    // generator, so simply calling location() directly won't work.
    auto loc =
        dynamic_cast<MacroniMetaGenerator &>(derived().meta).location(*sub);
    auto name_tok = sub->NameOrOperator();
    auto macro_name = (name_tok ? name_tok->Data() : "<a nameless macro>");
    auto function_like = is_sub_function_like(*sub);
    auto parameter_names = get_parameter_names(*sub);

    // We call `make_stmt_expr_region` here because a macro may not expand to an
    // expression
    auto [region, return_type] = make_maybe_value_yield_region(stmt);

    // Check if the macro is an expansion or a parameter, and return the
    // appropriate operation
    auto macroni_op = [&]() -> mlir::Operation * {
      if (sub->Kind() == pasta::MacroKind::kExpansion) {
        return stmt_visitor::template make<macroni::MacroExpansion>(
            loc, mlir_builder().getStringAttr(llvm::Twine(macro_name)),
            mlir_builder().getStrArrayAttr(llvm::ArrayRef(parameter_names)),
            mlir_builder().getBoolAttr(function_like), return_type,
            std::move(region));
      }
      return stmt_visitor::template make<macroni::MacroParameter>(
          loc, mlir_builder().getStringAttr(llvm::Twine(macro_name)),
          return_type, std::move(region));
    };
    mlir::Operation *op = macroni_op();
    derived().AlignedStmtVisited(pasta_stmt, *sub, op);
    return op;
  }

  // Hook called whenever Macroni finishes visiting a Stmt that does not align
  // with a macro substitution.
  // \param pasta_stmt  The `pasta::Stmt` that does not align with a macro
  // substitution.
  // \param op          The `mlir::Operation` obtained from visiting the `Stmt`.
  void UnalignedStmtVisited(pasta::Stmt &pasta_stmt, mlir::Operation *op) {}

  // Hook called whenever Macroni finishes visiting a Stmt that align with a
  // macro substitution.
  // \param pasta_stmt    The `pasta::Stmt` that aligns with a macro
  // substitution.
  // \param sub           The `pasta::MacroSubstitution` that `pasta_stmt`
  // aligns with.
  // \param op The `mlir::Operation` obtained from visiting the `Stmt`.
  void AlignedStmtVisited(pasta::Stmt &pasta_stmt,
                          pasta::MacroSubstitution &sub, mlir::Operation *op) {}
};

template <typename Derived>
using MacroniVisitorConfig =
    vast::cg::fallback_visitor<Derived, MacroniCodeGenVisitorMixin,
                               vast::cg::unsup_visitor,
                               vast::cg::unreach_visitor>;

using MacroniCodeGenInstance = vast::cg::codegen_instance<MacroniVisitorConfig>;

} // namespace macroni
