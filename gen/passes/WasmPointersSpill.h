#pragma once
#include "gen/llvm.h"
#include "gen/passes/Passes.h"

struct LLVM_LIBRARY_VISIBILITY WasmPointersSpill {
  bool run(llvm::Function &F);

  static llvm::StringRef getPassName() { return "WasmPointersSpill"; }
};

struct LLVM_LIBRARY_VISIBILITY WasmPointersSpillPass : public llvm::PassInfoMixin<WasmPointersSpillPass> {
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &fam) {
    if (pass.run(F)) {
     return llvm::PreservedAnalyses::none();
    }
    else {
     return llvm::PreservedAnalyses::all();
    }
  }
  static llvm::StringRef name() { return WasmPointersSpill::getPassName(); }

  WasmPointersSpillPass() : pass() {}
private:
  WasmPointersSpill pass;
};
