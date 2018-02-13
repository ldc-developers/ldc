#ifndef PGO_H
#define PGO_H

#include <string>
#include <unordered_map>

#include <llvm/ADT/STLExtras.h>

struct Context;

namespace llvm {
class Module;
class PassManagerBuilder;
}

class PgoHandler final {
  std::string Filename;
public:
  PgoHandler(const Context &context,
             llvm::Module &module,
             std::unordered_map<std::string, void *> &symbols,
             llvm::PassManagerBuilder &builder);
  ~PgoHandler();
};

void bindPgoSymbols(const Context& context,
                    llvm::function_ref<void*(llvm::StringRef)> getter);

#endif // PGO_H
