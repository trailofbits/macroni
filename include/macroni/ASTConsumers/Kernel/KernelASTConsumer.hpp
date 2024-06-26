#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>

namespace macroni::kernel {
class KernelASTConsumer : public clang::ASTConsumer {
public:
  KernelASTConsumer(bool print_locations);

  void HandleTranslationUnit(clang::ASTContext &Ctx) override;

private:
  bool m_print_locations;
};

class KernelASTConsumerFactory {
public:
  std::unique_ptr<KernelASTConsumer> newASTConsumer(void);

  KernelASTConsumerFactory(bool print_locations);

private:
  bool m_print_locations;
};
} // namespace macroni::kernel