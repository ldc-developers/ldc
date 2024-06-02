#pragma once
#include "gen/llvm.h"
#include "gen/passes/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/AliasAnalysis.h"

//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
class LLVM_LIBRARY_VISIBILITY LibCallOptimization {
protected:
  llvm::Function *Caller;
  bool *Changed;
  const llvm::DataLayout *DL;
  llvm::AliasAnalysis *AA;
  llvm::LLVMContext *Context;

  /// EmitMemCpy - Emit a call to the memcpy function to the builder.  This
  /// always expects that the size has type 'intptr_t' and Dst/Src are pointers.
  llvm::Value *EmitMemCpy(llvm::Value *Dst, llvm::Value *Src, llvm::Value *Len, unsigned Align,
                    IRBuilder<> &B);

public:
  LibCallOptimization() = default;
  virtual ~LibCallOptimization() = default;

  /// CallOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual llvm::Value *CallOptimizer(llvm::Function *Callee, llvm::CallInst *CI,
                               IRBuilder<> &B) = 0;

  llvm::Value *OptimizeCall(llvm::CallInst *CI, bool &Changed, const llvm::DataLayout *DL,
                      llvm::AliasAnalysis &AA, llvm::IRBuilder<> &B);
};

/// ArraySetLengthOpt - remove libcall for arr.length = N if N <= arr.length
struct LLVM_LIBRARY_VISIBILITY ArraySetLengthOpt : public LibCallOptimization {
  llvm::Value *CallOptimizer(llvm::Function *Callee, llvm::CallInst *CI,
                       llvm::IRBuilder<> &B) override; 
};
/// AllocationOpt - Common optimizations for various GC allocations.
struct LLVM_LIBRARY_VISIBILITY AllocationOpt : public LibCallOptimization {
  llvm::Value *CallOptimizer(llvm::Function *Callee, llvm::CallInst *CI,
                       llvm::IRBuilder<> &B) override;
};
/// ArraySliceCopyOpt - Turn slice copies into llvm.memcpy when safe
struct LLVM_LIBRARY_VISIBILITY ArraySliceCopyOpt : public LibCallOptimization {
  llvm::Value *CallOptimizer(llvm::Function *Callee, llvm::CallInst *CI,
                       llvm::IRBuilder<> &B) override;
 
};

/// This pass optimizes library functions from the D runtime as used by LDC.
///
struct LLVM_LIBRARY_VISIBILITY SimplifyDRuntimeCalls {
  llvm::StringMap<LibCallOptimization *> Optimizations;

  // Array operations
  ArraySetLengthOpt ArraySetLength;
  ArraySliceCopyOpt ArraySliceCopy;

  // GC allocations
  AllocationOpt Allocation;

  void InitOptimizations();
  bool run(llvm::Function &F, std::function<llvm::AAResults& ()>  getAA);

  bool runOnce(llvm::Function &F, const llvm::DataLayout *DL, llvm::AAResults &AA);
  static llvm::StringRef getPassName() { return "SimplifyDRuntimeCalls"; }
};

struct LLVM_LIBRARY_VISIBILITY SimplifyDRuntimeCallsPass : public llvm::PassInfoMixin<SimplifyDRuntimeCallsPass> {

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &fam) {
    auto getAA = [&]() -> llvm::AAResults& {
      return fam.getResult<llvm::AAManager>(F);
    };

    if (pass.run(F, getAA)) {
     return llvm::PreservedAnalyses::none();
    }
    else {
     return llvm::PreservedAnalyses::all();
    }
  }
  static llvm::StringRef name() { return SimplifyDRuntimeCalls::getPassName(); }

  SimplifyDRuntimeCallsPass() : pass() {}
private:
  SimplifyDRuntimeCalls pass;
};
