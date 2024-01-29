#include <array>
#include <cstddef>
#include <macroni/Conversion/Kernel/KernelRewriters.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace macroni::kernel {

template <class T> bool has_name(macroni::MacroExpansion exp) {
  return exp.getMacroName() == T::getOperationName().split('.').second;
}

bool has_results_and_n_parameters(macroni::MacroExpansion exp, size_t n) {
  return exp.getNumResults() > 0 && exp.getParameterNames().size() == n;
}

bool is_get_user(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 2) && has_name<GetUser>(exp);
}

bool is_offsetof(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 2) && has_name<OffsetOf>(exp);
}

bool is_container_of(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 3) && has_name<ContainerOf>(exp);
}

bool is_rcu_dereference(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 1) &&
         (has_name<RCUDereference>(exp) || has_name<RCUDereferenceBH>(exp) ||
          has_name<RCUDereferenceSched>(exp));
}

bool is_rcu_dereference_check(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 2) &&
         (has_name<RCUDereferenceCheck>(exp) ||
          has_name<RCUDereferenceBHCheck>(exp) ||
          has_name<RCUDereferenceSchedCheck>(exp) ||
          has_name<RCUDereferenceProtected>(exp));
}

bool is_rcu_access_pointer(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 1) &&
         has_name<RCUAccessPointer>(exp);
}

bool is_rcu_assign_pointer(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 2) &&
         has_name<RCUAssignPointer>(exp);
}

bool is_rcu_replace_pointer(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 3) &&
         has_name<RCUReplacePointer>(exp);
}

bool is_smp_mb(macroni::MacroExpansion exp) {
  return has_results_and_n_parameters(exp, 0) && has_name<SMPMB>(exp);
}

template <typename... Args>
std::array<mlir::Operation *, sizeof...(Args)>
fetch_macro_parameters(mlir::Operation *op, Args &&...args) {
  auto results = std::array<mlir::Operation *, sizeof...(Args)>();
  auto names = std::array<const char *, sizeof...(Args)>({args...});
  auto walker = [&](macroni::MacroParameter mp) {
    for (size_t i = 0; i < sizeof...(Args); i++) {
      if (results[i] == nullptr && mp.getParameterName().equals(names[i])) {
        results[i] = mp.getOperation();
      }
    }
  };
  op->walk(walker);
  return results;
}

template <typename T, typename... ReplacementArgs>
bool replace_if_name_matches(mlir::PatternRewriter &rewriter,
                             macroni::MacroExpansion &exp,
                             ReplacementArgs &&...replacement_args) {
  if (!has_name<T>(exp)) {
    return false;
  }
  rewriter.replaceOpWithNewOp<T>(
      exp, exp.getType(0), std::forward<ReplacementArgs>(replacement_args)...);
  return true;
}

mlir::LogicalResult rewrite_get_user(macroni::MacroExpansion exp,
                                     mlir::PatternRewriter &rewriter) {
  if (!is_get_user(exp)) {
    return mlir::failure();
  }
  auto [x, ptr] = fetch_macro_parameters(exp, "x", "ptr");
  if (!(x && ptr && x->getNumResults() == 1 && ptr->getNumResults() == 1)) {
    return mlir::failure();
  }

  // The x and ptr macro parameters are expanded inside of the
  // expansion of get_user(), so if we simply replaced the original
  // macro expansion with the GetUser kernel operation, we would lose
  // the arguments. Therefore we clone the arguments to lift them out
  // of the macro expansion, and pass them to the GetUser operation.
  // NOTE(bpp): Maybe we should create an operation for lifted macro
  // parameters to make it possible to distinguish normal macro
  // parameters from ones that have been lifted?
  auto x_clone = rewriter.clone(*x);
  auto ptr_clone = rewriter.clone(*ptr);
  rewriter.replaceOpWithNewOp<GetUser>(
      exp, exp.getType(0), x_clone->getResult(0), ptr_clone->getResult(0));
  return mlir::success();
}

mlir::LogicalResult rewrite_offsetof(macroni::MacroExpansion exp,
                                     mlir::PatternRewriter &rewriter) {
  if (!is_offsetof(exp)) {
    return mlir::failure();
  }
  std::optional<mlir::TypeAttr> type;
  std::optional<mlir::StringAttr> member;
  auto walker = [&](mlir::Operation *op) {
    auto member_op = mlir::dyn_cast<vast::hl::RecordMemberOp>(op);
    if (!member_op) {
      return;
    }
    auto record_ty = member_op.getRecord().getType();
    if (auto pty = mlir::dyn_cast<vast::hl::PointerType>(record_ty)) {
      type = mlir::TypeAttr::get(pty.getElementType());
    } else {
      type = mlir::TypeAttr::get(record_ty);
    }
    member = member_op.getNameAttr();
  };
  exp.getExpansion().walk(walker);
  if (!(type && member)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<OffsetOf>(exp, exp.getType(0), *type, *member);
  return mlir::success();
}

mlir::LogicalResult rewrite_container_of(macroni::MacroExpansion exp,
                                         mlir::PatternRewriter &rewriter) {
  if (!is_container_of(exp)) {
    return mlir::failure();
  }
  mlir::Operation *ptr = nullptr;
  std::optional<mlir::TypeAttr> type;
  std::optional<mlir::StringAttr> member;
  auto walker = [&](mlir::Operation *op) {
    if (auto mp = mlir::dyn_cast<macroni::MacroParameter>(op);
        mp && mp.getParameterName() == "ptr") {
      ptr = op;
      return;
    }
    auto member_op = mlir::dyn_cast<vast::hl::RecordMemberOp>(op);
    if (!member_op) {
      return;
    }
    auto record_ty = member_op.getRecord().getType();
    if (auto pty = mlir::dyn_cast<vast::hl::PointerType>(record_ty)) {
      type = mlir::TypeAttr::get(pty.getElementType());
    } else {
      type = mlir::TypeAttr::get(record_ty);
    }
    member = member_op.getNameAttr();
  };
  exp.getExpansion().walk(walker);
  if (!(ptr && type && member && ptr->getNumResults() == 1)) {
    return mlir::failure();
  }

  auto ptr_clone = rewriter.clone(*ptr);
  rewriter.replaceOpWithNewOp<ContainerOf>(
      exp, exp.getType(0), ptr_clone->getResult(0), *type, *member);
  return mlir::success();
}

mlir::LogicalResult rewrite_rcu_dereference(macroni::MacroExpansion exp,
                                            mlir::PatternRewriter &rewriter) {
  if (!is_rcu_dereference(exp)) {
    return mlir::failure();
  }
  auto [p] = fetch_macro_parameters(exp, "p");
  if (!(p && p->getNumResults() == 1)) {
    return mlir::failure();
  }

  auto p_clone = rewriter.clone(*p);
  auto p_clone_result = p_clone->getResult(0);
  return mlir::success(
      replace_if_name_matches<RCUDereference>(rewriter, exp, p_clone_result) ||
      replace_if_name_matches<RCUDereferenceBH>(rewriter, exp,
                                                p_clone_result) ||
      replace_if_name_matches<RCUDereferenceSched>(rewriter, exp,
                                                   p_clone_result));
}

mlir::LogicalResult
rewrite_rcu_dereference_check(macroni::MacroExpansion exp,
                              mlir::PatternRewriter &rewriter) {
  if (!is_rcu_dereference_check(exp)) {
    return mlir::failure();
  }
  auto [p, c] = fetch_macro_parameters(exp, "p", "c");
  if (!((p && p->getNumResults() == 1) && (c && c->getNumResults() == 1))) {
    return mlir::failure();
  }

  auto p_clone = rewriter.clone(*p);
  auto c_clone = rewriter.clone(*c);
  auto p_clone_result = p_clone->getResult(0);
  auto c_clone_result = c_clone->getResult(0);
  return mlir::success(replace_if_name_matches<RCUDereferenceCheck>(
                           rewriter, exp, p_clone_result, c_clone_result) ||
                       replace_if_name_matches<RCUDereferenceBHCheck>(
                           rewriter, exp, p_clone_result, c_clone_result) ||
                       replace_if_name_matches<RCUDereferenceSchedCheck>(
                           rewriter, exp, p_clone_result, c_clone_result) ||
                       replace_if_name_matches<RCUDereferenceProtected>(
                           rewriter, exp, p_clone_result, c_clone_result));
}

mlir::LogicalResult
rewrite_rcu_access_pointer(macroni::MacroExpansion exp,
                           mlir::PatternRewriter &rewriter) {
  if (!is_rcu_access_pointer(exp)) {
    return mlir::failure();
  }
  auto [p] = fetch_macro_parameters(exp, "p");
  if (!(p && p->getNumResults() == 1)) {
    return mlir::failure();
  }

  auto p_clone = rewriter.clone(*p);
  rewriter.replaceOpWithNewOp<RCUAccessPointer>(exp, exp.getType(0),
                                                p_clone->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
rewrite_rcu_assign_pointer(macroni::MacroExpansion exp,
                           mlir::PatternRewriter &rewriter) {
  if (!is_rcu_assign_pointer(exp)) {
    return mlir::failure();
  }
  auto [p, v] = fetch_macro_parameters(exp, "p", "v");
  if (!((p && p->getNumResults() == 1) && (v && v->getNumResults() == 1))) {
    return mlir::failure();
  }

  auto p_clone = rewriter.clone(*p);
  auto v_clone = rewriter.clone(*v);
  rewriter.replaceOpWithNewOp<RCUAssignPointer>(
      exp, exp.getType(0), p_clone->getResult(0), v_clone->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
rewrite_rcu_replace_pointer(macroni::MacroExpansion exp,
                            mlir::PatternRewriter &rewriter) {
  if (!is_rcu_replace_pointer(exp)) {
    return mlir::failure();
  }
  auto [rcu_ptr, ptr, c] = fetch_macro_parameters(exp, "rcu_ptr", "ptr", "c");
  if (!((rcu_ptr && rcu_ptr->getNumResults() == 1) &&
        (ptr && ptr->getNumResults() == 1) && (c && c->getNumResults() == 1))) {
    return mlir::failure();
  }

  auto rcu_ptr_clone = rewriter.clone(*rcu_ptr);
  auto ptr_clone = rewriter.clone(*ptr);
  auto c_clone = rewriter.clone(*c);
  rewriter.replaceOpWithNewOp<RCUReplacePointer>(
      exp, exp.getType(0), rcu_ptr_clone->getResult(0), ptr_clone->getResult(0),
      c_clone->getResult(0));
  return mlir::success();
}

mlir::LogicalResult rewrite_smp_mb(macroni::MacroExpansion exp,
                                   mlir::PatternRewriter &rewriter) {
  if (!is_smp_mb(exp)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<kernel::SMPMB>(exp, exp.getType(0));
  return mlir::success();
}

mlir::LogicalResult rewrite_list_for_each(vast::hl::ForOp for_op,
                                          mlir::PatternRewriter &rewriter) {
  auto [pos, head] = fetch_macro_parameters(for_op, "pos", "head");
  if (!(pos && head)) {
    return mlir::failure();
  }

  auto pos_clone = rewriter.clone(*pos);
  auto head_clone = rewriter.clone(*head);
  auto reg = std::make_unique<mlir::Region>();
  reg->takeBody(for_op.getBodyRegion());
  rewriter.replaceOpWithNewOp<ListForEach>(for_op, pos_clone->getResult(0),
                                           head_clone->getResult(0),
                                           std::move(reg));
  return mlir::success();
}

llvm::APInt get_lock_level(mlir::Operation &op) {
  return op.getAttrOfType<mlir::IntegerAttr>("lock_level").getValue();
}

mlir::LogicalResult rewrite_rcu_read_unlock(vast::hl::CallOp call_op,
                                            mlir::PatternRewriter &rewriter) {
  auto name = call_op.getCalleeAttr().getValue();
  if ("rcu_read_unlock" != name) {
    return mlir::failure();
  }
  auto unlock_op = call_op.getOperation();
  auto unlock_level = get_lock_level(*unlock_op);
  mlir::Operation *lock_op = nullptr;
  for (auto op = unlock_op; op; op = op->getPrevNode()) {
    if (auto other_call_op = mlir::dyn_cast<vast::hl::CallOp>(op)) {
      name = other_call_op.getCalleeAttr().getValue();
      if ("rcu_read_lock" == name) {
        auto lock_level = get_lock_level(*op);
        if (unlock_level == lock_level) {
          lock_op = op;
          break;
        }
      }
    }
  }
  if (!lock_op) {
    return mlir::failure();
  }

  rewriter.setInsertionPointAfter(lock_op);
  auto cs = rewriter.replaceOpWithNewOp<RCUCriticalSection>(lock_op);
  auto cs_block = rewriter.createBlock(&cs.getBodyRegion());
  for (auto op = cs->getNextNode(); op != unlock_op;) {
    auto temp = op->getNextNode();
    op->moveBefore(cs_block, cs_block->end());
    op = temp;
  }
  rewriter.eraseOp(unlock_op);
  return mlir::success();
}

mlir::LogicalResult rewrite_label_stmt(vast::hl::LabelStmt label_stmt,
                                       mlir::PatternRewriter &rewriter) {
  // In the Linux Kernel, a common idiom is to call `rcu_read_unlock()` right
  // after declaring a label. This idiom prevents our `CallOp` pattern rewriter
  // from detecting such unlocks as the end of a critical section. To fix this,
  // we look for labels which are immediately followed by a call to
  // `rcu_read_unlock()`, and move the call before the label.

  auto ops = label_stmt.getOps();
  if (ops.empty()) {
    return mlir::failure();
  }

  auto call_op = mlir::dyn_cast<vast::hl::CallOp>(*ops.begin());
  if (!call_op || call_op.getCalleeAttr().getValue() != "rcu_read_unlock") {
    return mlir::failure();
  }

  // TODO(bpp): Keep track of which calls to `rcu_read_unlock()` were originally
  // nested under labels.
  rewriter.setInsertionPoint(label_stmt);
  rewriter.clone(*call_op.getOperation());
  rewriter.eraseOp(call_op);
  return mlir::success();
}

} // namespace macroni::kernel
