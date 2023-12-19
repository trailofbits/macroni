#include <macroni/Common/ParseAST.hpp>

namespace pasta {
Result<AST, std::string> parse_ast(int argc, char **argv) {
  InitPasta initializer;
  FileManager fm(FileSystem::CreateNative());
  auto maybe_compiler = Compiler::CreateHostCompiler(fm, TargetLanguage::kC);
  if (!maybe_compiler.Succeeded()) {
    return maybe_compiler.TakeError();
  }

  auto compiler = maybe_compiler.Value();
  auto fs = FileSystem::From(compiler);
  auto maybe_cwd = fs->CurrentWorkingDirectory();
  if (!maybe_cwd.Succeeded()) {
    return maybe_compiler.TakeError();
  }

  auto cwd = maybe_cwd.TakeValue();
  const ArgumentVector args(argc - 1, &argv[1]);
  auto maybe_command = CompileCommand::CreateFromArguments(args, cwd);
  if (!maybe_command.Succeeded()) {
    return std::string(maybe_command.TakeError());
  }

  const auto command = maybe_command.TakeValue();
  auto maybe_jobs = compiler.CreateJobsForCommand(command);
  if (!maybe_jobs.Succeeded()) {
    return maybe_jobs.TakeError();
  }

  auto jobs = maybe_jobs.TakeValue();
  auto maybe_ast = jobs.front().Run();
  if (!maybe_ast.Succeeded()) {
    return maybe_ast.TakeError();
  }

  return maybe_ast.TakeValue();
}
} // namespace pasta
