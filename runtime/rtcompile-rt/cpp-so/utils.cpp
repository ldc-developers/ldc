#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "context.h"


void fatal(const Context &context, const std::string &reason)
{
    if (nullptr != context.fatalHandler) {
        context.fatalHandler(context.fatalHandlerData,
                             reason.c_str());
    }
    else {
        fprintf(stderr, "Runtime compiler fatal: %s\n", reason.c_str());
        abort();
    }
}

void interruptPoint(const Context &context, const char *desc, const char *object)
{
    assert(nullptr != desc);
    if (nullptr != context.interruptPointHandler) {
      context.interruptPointHandler(context.interruptPointHandlerData,
                                    desc,
                                    object);
    }
}

void verifyModule(const Context &context, llvm::Module &module)
{
  std::string err;
  llvm::raw_string_ostream errstream(err);
  if (llvm::verifyModule(module, &errstream)) {
    std::string desc = std::string("module verification failed:") +
                       errstream.str();
    fatal(context, desc);
  }
}
