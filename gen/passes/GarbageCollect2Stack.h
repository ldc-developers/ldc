#pragma once
#include "gen/llvm.h"
#include "gen/passes/Passes.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"

struct G2StackAnalysis;
 
//===----------------------------------------------------------------------===//
// Helpers for specific types of GC calls.
//===----------------------------------------------------------------------===//

//namespace {
namespace ReturnType {
enum Type {
  Pointer, /// Function returns a pointer to the allocated memory.
  Array    /// Function returns the allocated memory as an array slice.
};
}
class FunctionInfo {
protected:
  llvm::Type *Ty;

public:
  ReturnType::Type ReturnType;

  // Analyze the current call, filling in some fields. Returns true if
  // this is an allocation we can stack-allocate.
  virtual bool analyze(llvm::CallBase *CB, const G2StackAnalysis &A) = 0;

  // Returns the alloca to replace this call.
  // It will always be inserted before the call.
  virtual llvm::Value *promote(llvm::CallBase *CB, IRBuilder<> &B, const G2StackAnalysis &A);

  explicit FunctionInfo(ReturnType::Type returnType) : ReturnType(returnType) {}
  virtual ~FunctionInfo() = default;
};
class TypeInfoFI : public FunctionInfo {
public:
  unsigned TypeInfoArgNr;

  TypeInfoFI(ReturnType::Type returnType, unsigned tiArgNr)
      : FunctionInfo(returnType), TypeInfoArgNr(tiArgNr) {}

  bool analyze(llvm::CallBase *CB, const G2StackAnalysis &A) override;
};
class ArrayFI : public TypeInfoFI {
  int ArrSizeArgNr;
  bool Initialized;
  llvm::Value *arrSize;

public:
  ArrayFI(ReturnType::Type returnType, unsigned tiArgNr, unsigned arrSizeArgNr,
          bool initialized)
      : TypeInfoFI(returnType, tiArgNr), ArrSizeArgNr(arrSizeArgNr),
        Initialized(initialized) {}

  bool analyze(llvm::CallBase *CB, const G2StackAnalysis &A) override;

  llvm::Value *promote(llvm::CallBase *CB, IRBuilder<> &B, const G2StackAnalysis &A) override;

};
// FunctionInfo for _d_allocclass
class AllocClassFI : public FunctionInfo {
public:
  bool analyze(llvm::CallBase *CB, const G2StackAnalysis &A) override;

  // The default promote() should be fine.

  AllocClassFI() : FunctionInfo(ReturnType::Pointer) {}
};
/// Describes runtime functions that allocate a chunk of memory with a
/// given size.
class UntypedMemoryFI : public FunctionInfo {
  unsigned SizeArgNr;
  llvm::Value *SizeArg;

public:
  bool analyze(llvm::CallBase *CB, const G2StackAnalysis &A) override;

  llvm::Value *promote(llvm::CallBase *CB, IRBuilder<> &B, const G2StackAnalysis &A) override;

  explicit UntypedMemoryFI(unsigned sizeArgNr)
      : FunctionInfo(ReturnType::Pointer), SizeArgNr(sizeArgNr) {}
};
//}

//===----------------------------------------------------------------------===//
// GarbageCollect2Stack Pass Implementation
//===----------------------------------------------------------------------===//

//namespace {
/// This pass replaces GC calls with alloca's
///
struct GarbageCollect2Stack {
  llvm::Module *M;

  TypeInfoFI AllocMemoryT;
  ArrayFI NewArrayU;
  ArrayFI NewArrayT;
  AllocClassFI AllocClass;
  UntypedMemoryFI AllocMemory;

  GarbageCollect2Stack();

  bool run(llvm::Function& function,
           std::function<llvm::DominatorTree& ()> getDT,
           std::function<llvm::CallGraph * ()> getCG);

  static llvm::StringRef getPassName() { return "GarbageCollect2Stack"; }
};
struct LLVM_LIBRARY_VISIBILITY GarbageCollect2StackPass : public llvm::PassInfoMixin<GarbageCollect2StackPass> {

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &fam) {

    auto getDT = [&]() -> llvm::DominatorTree& {
      return fam.getResult<llvm::DominatorTreeAnalysis>(F);
    };

    auto getCG = [&]() -> llvm::CallGraph* {
//FIXME: Seems like you can't do this but how else can we get the call graph?
//      const auto &mamp =  fam.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
//      return F.getParent() == nullptr ? nullptr
//                                      : mamp.getCachedResult<llvm::CallGraphAnalysis>(*F.getParent());
// 
      return nullptr;
    };

    if (pass.run(F, getDT, getCG)) {
      llvm::PreservedAnalyses pa;
      pa.preserve<llvm::CallGraphAnalysis>();
      return pa;
    }
    else {
     return llvm::PreservedAnalyses::all();
    }
  }
  static llvm::StringRef name() { return GarbageCollect2Stack::getPassName(); }

  GarbageCollect2StackPass() : pass() {}
private:
  GarbageCollect2Stack pass;
};
//}
