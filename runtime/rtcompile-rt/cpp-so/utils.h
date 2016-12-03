#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

struct Context;
namespace llvm {
class Module;
}

void fatal(const Context &context, const std::string &reason);
void interruptPoint(const Context &context, const char *desc, const char *object = "");
void verifyModule(const Context &context, llvm::Module &module);

#endif // UTILS_HPP
