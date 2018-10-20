//===-- StripExternals.cpp - Strip available_externally symbols -----------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This transform strips the bodies of available_externally functions and
// initializers of available_externally globals, turning them into external
// declarations.
// This is useful to allow Global DCE (-globaldce) to clean up references to
// globals only used by available_externally functions and initializers.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strip-externals"
#if LDC_LLVM_VER < 700
#define LLVM_DEBUG DEBUG
#endif

#include "gen/passes/Passes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumFunctions, "Number of function bodies removed");
STATISTIC(NumVariables, "Number of global initializers removed");

namespace {
struct LLVM_LIBRARY_VISIBILITY StripExternals : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  StripExternals() : ModulePass(ID) {}

  // run - Do the StripExternals pass on the specified module.
  //
  bool runOnModule(Module &M) override;
};
}

char StripExternals::ID = 0;
static RegisterPass<StripExternals>
    X("strip-externals", "Strip available_externally bodies and initializers");

ModulePass *createStripExternalsPass() { return new StripExternals(); }

bool StripExternals::runOnModule(Module &M) {
  bool Changed = false;

  for (auto I = M.begin(); I != M.end();) {
    if (I->hasAvailableExternallyLinkage()) {
      assert(!I->isDeclaration() &&
             "Declarations can't be available_externally");
      Changed = true;
      ++NumFunctions;
      if (I->use_empty()) {
        LLVM_DEBUG(errs() << "Deleting function: " << *I);
        auto todelete = I;
        ++I;
        todelete->eraseFromParent();
        continue;
      } else {
        I->deleteBody();
        LLVM_DEBUG(errs() << "Deleted function body: " << *I);
      }
    }
    ++I;
  }

  for (auto I = M.global_begin(); I != M.global_end();) {
    if (I->hasAvailableExternallyLinkage()) {
      assert(!I->isDeclaration() &&
             "Declarations can't be available_externally");
      Changed = true;
      ++NumVariables;
      if (I->use_empty()) {
        LLVM_DEBUG(errs() << "Deleting global: " << *I);
        auto todelete = I;
        ++I;
        todelete->eraseFromParent();
        continue;
      } else {
        I->setInitializer(nullptr);
        I->setLinkage(GlobalValue::ExternalLinkage);
        LLVM_DEBUG(errs() << "Deleted initializer: " << *I);
      }
    }
    ++I;
  }

  return Changed;
}
