#pragma once

#include "macroni/Common/Common.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>

namespace macroni::kernel {
class KernelASTConsumer : public clang::ASTConsumer {
public:
  KernelASTConsumer(module_handler mod_handler);

  void HandleTranslationUnit(clang::ASTContext &Ctx) override;

private:
  module_handler m_module_handler;
};

class KernelASTConsumerFactory {
public:
  std::unique_ptr<KernelASTConsumer> newASTConsumer(void);

  KernelASTConsumerFactory(module_handler mod_handler);

private:
  module_handler m_module_handler;
};
} // namespace macroni::kernel