//===-- addFuncEntryCallPass.cpp - Optimize druntime calls ----------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the University of Illinois Open Source
// License. See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

namespace {

class FuncEntryCallPass : public FunctionPass {

  Value *funcToCallUponEntry = nullptr;

public:
  static char ID;
  FuncEntryCallPass() : FunctionPass(ID) {}

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
};
}

char FuncEntryCallPass::ID = 0;

bool FuncEntryCallPass::doInitialization(Module &M) {
  // Add fwd declaration of the `void __test_funcentrycall(void)` function.
  auto functionType = FunctionType::get(Type::getVoidTy(M.getContext()), false);
  funcToCallUponEntry =
      M.getOrInsertFunction("__test_funcentrycall", functionType)
#if LLVM_VERSION >= 900
          .getCallee()
#endif
      ;
  return true;
}

bool FuncEntryCallPass::runOnFunction(Function &F) {
  // Add call to `__test_funcentrycall(void)` at the start of _every_ function
  // (this includes e.g. `ldc.register_dso`!)
  llvm::BasicBlock &block = F.getEntryBlock();
  IRBuilder<> builder(&block, block.begin());
#if LLVM_VERSION >= 1100
  builder.CreateCall(FunctionCallee(cast<Function>(funcToCallUponEntry)));
#else
  builder.CreateCall(funcToCallUponEntry);
#endif
  return true;
}

static void addFuncEntryCallPass(const PassManagerBuilder &,
                                 legacy::PassManagerBase &PM) {
  PM.add(new FuncEntryCallPass());
}
// Registration of the plugin's pass is done by the plugin's static constructor.
static RegisterStandardPasses
    RegisterFuncEntryCallPass0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                               addFuncEntryCallPass);
