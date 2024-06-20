#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>

namespace macroni::kernel {
class KernelASTConsumer : public clang::ASTConsumer {
public:
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};

class KernelASTConsumerFactory {
public:
  std::unique_ptr<KernelASTConsumer> newASTConsumer(void);
};
} // namespace macroni::kernel