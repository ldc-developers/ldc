#ifndef PGO_H
#define PGO_H

#include <string>
#include <unordered_map>

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

#endif // PGO_H
