#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>

namespace macroni::safety {
class SafetyASTConsumer : public clang::ASTConsumer {
public:
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};

class SafetyASTConsumerFactory {
public:
  std::unique_ptr<SafetyASTConsumer> newASTConsumer(void);
};
} // namespace macroni::safety