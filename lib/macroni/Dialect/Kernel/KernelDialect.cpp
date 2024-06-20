// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Common/MacroSpelling.hpp"
#include "macroni/Dialect/Kernel/KernelAttributes.hpp"
#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "macroni/Dialect/Kernel/KernelTypes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeSupport.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>

namespace macroni::kernel {
const macro_spelling KernelDialect::rcu_access_pointer =
    macro_spelling{"rcu_access_pointer", {"p"}};
const macro_spelling KernelDialect::rcu_assign_pointer =
    macro_spelling{"rcu_assign_pointer", {"p", "v"}};
const macro_spelling KernelDialect::rcu_dereference =
    macro_spelling{"rcu_dereference", {"p"}};
const macro_spelling KernelDialect::rcu_dereference_bh =
    macro_spelling{"rcu_dereference_bh", {"p"}};
const macro_spelling KernelDialect::rcu_dereference_sched =
    macro_spelling{"rcu_dereference_sched", {"p"}};
const macro_spelling KernelDialect::rcu_replace_pointer =
    macro_spelling{"rcu_replace_pointer", {"rcu_ptr", "ptr", "c"}};

const std::vector<macro_spelling> KernelDialect::rcu_macro_spellings = {
    KernelDialect::rcu_access_pointer,    KernelDialect::rcu_assign_pointer,
    KernelDialect::rcu_dereference,       KernelDialect::rcu_dereference_bh,
    KernelDialect::rcu_dereference_sched, KernelDialect::rcu_replace_pointer,
};

void KernelDialect::initialize() {
  registerAttributes();
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "macroni/Dialect/Kernel/Kernel.cpp.inc"
      >();
}

using DialectParser = mlir::AsmParser;
using DialectPrinter = mlir::AsmPrinter;

} // namespace macroni::kernel

#include "macroni/Dialect/Kernel/KernelDialect.cpp.inc"