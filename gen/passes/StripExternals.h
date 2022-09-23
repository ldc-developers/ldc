#pragma once
#include "gen/llvm.h"
#include "gen/passes/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"
struct LLVM_LIBRARY_VISIBILITY StripExternals {
  // run - Do the StripExternals pass on the specified module.
  //
  bool run(llvm::Module &M);
};

struct LLVM_LIBRARY_VISIBILITY StripExternalsPass : public llvm::PassInfoMixin<StripExternalsPass> {

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &mam) {

    if (pass.run(M)) {
      //FIXME: What is preserved?
      return llvm::PreservedAnalyses::none();
    }
    else {
      return llvm::PreservedAnalyses::all();
    }
  }

  static llvm::StringRef name() { return "StripExternals"; }

  StripExternalsPass() : pass() {}
private:
  StripExternals pass;
};
