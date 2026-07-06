//===-- StripStackGCRoot.cpp - Strip _d_stack_gcroot calls -----------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This transform strips all calls to _d_stack_gcroot.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strip-stack-gcroot"

#include "gen/passes/Passes.h"
#include "gen/passes/StripStackGCRoot.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

STATISTIC(NumCalls, "Number of calls removed");

struct LLVM_LIBRARY_VISIBILITY StripStackGCRootLegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  StripStackGCRootLegacyPass() : FunctionPass(ID) {}
  StripStackGCRoot pass;
  // run - Do the StripStackGCRoot pass on the specified module.
  //
    bool runOnFunction(Function &F) override { return pass.run(F); };
};

char StripStackGCRootLegacyPass::ID = 0;
static RegisterPass<StripStackGCRootLegacyPass>
    X("strip-stack-gcroot", "Strip calls to _d_stack_gcroot");

FunctionPass *createStripStackGCRootPass() { return new StripStackGCRootLegacyPass(); }

bool StripStackGCRoot::run(Function &F) {
  bool Changed = false;

  for (auto &BB : F) {
    for (auto I = BB.begin(); I != BB.end();) {
      // Ignore non-calls.
      CallInst *CI = dyn_cast<CallInst>(&(*(I++)));
      if (!CI) {
        continue;
      }

      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CI->getCalledFunction();
      if (Callee == nullptr || !Callee->isDeclaration() ||
          !Callee->hasExternalLinkage()) {
        continue;
      }

      if (Callee->getName() != "_d_stack_gcroot") {
        continue;
      }


      CI->eraseFromParent();
      ++NumCalls;
      Changed = true;
    }
  }

  return Changed;
}
