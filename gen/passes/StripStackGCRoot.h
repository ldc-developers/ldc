#pragma once
#include "gen/llvm.h"
#include "gen/passes/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"
struct LLVM_LIBRARY_VISIBILITY StripStackGCRoot {
  // run - Do the StripExternals pass on the specified module.
  //
  bool run(llvm::Function &F);
};

struct LLVM_LIBRARY_VISIBILITY StripStackGCRootPass : public llvm::PassInfoMixin<StripStackGCRootPass> {

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &fam) {
    if (pass.run(F)) {
      //FIXME: What is preserved?
      return llvm::PreservedAnalyses::none();
    }
    else {
      return llvm::PreservedAnalyses::all();
    }
  }

  static llvm::StringRef name() { return "StripStackGCRoot"; }

  StripStackGCRootPass() : pass() {}
private:
  StripStackGCRoot pass;
};
