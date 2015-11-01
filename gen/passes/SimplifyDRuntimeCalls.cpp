//===-- SimplifyDRuntimeCalls.cpp - Optimize druntime calls ---------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the University of Illinois Open Source
// License. See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple pass that applies a variety of small
// optimizations for calls to specific functions in the D runtime.
//
// The machinery was copied from the standard -simplify-libcalls LLVM pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplify-drtcalls"

#include "Passes.h"
#include "llvm/Pass.h"
#if LDC_LLVM_VER >= 307
#include "llvm/IR/Module.h"
#endif
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "gen/runtime.h"

using namespace llvm;

#if LDC_LLVM_VER >= 308
typedef AAResultsWrapperPass AliasAnalysisPass;
#else
typedef AliasAnalysis AliasAnalysisPass;
#endif

STATISTIC(NumSimplified, "Number of runtime calls simplified");
STATISTIC(NumDeleted, "Number of runtime calls deleted");

//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class LLVM_LIBRARY_VISIBILITY LibCallOptimization {
protected:
  Function *Caller;
  bool *Changed;
  const DataLayout *DL;
  AliasAnalysis *AA;
  LLVMContext *Context;

  /// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
  Value *CastToCStr(Value *V, IRBuilder<> &B);

  /// EmitMemCpy - Emit a call to the memcpy function to the builder.  This
  /// always expects that the size has type 'intptr_t' and Dst/Src are pointers.
  Value *EmitMemCpy(Value *Dst, Value *Src, Value *Len, unsigned Align,
                    IRBuilder<> &B);

public:
  LibCallOptimization() {}
  virtual ~LibCallOptimization() {}

  /// CallOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI,
                               IRBuilder<> &B) = 0;

  Value *OptimizeCall(CallInst *CI, bool &Changed, const DataLayout *DL,
                      AliasAnalysis &AA, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->Changed = &Changed;
    this->DL = DL;
    this->AA = &AA;
    if (CI->getCalledFunction())
      Context = &CI->getCalledFunction()->getContext();
    return CallOptimizer(CI->getCalledFunction(), CI, B);
  }
};
} // End anonymous namespace.

/// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
Value *LibCallOptimization::CastToCStr(Value *V, IRBuilder<> &B) {
  return B.CreateBitCast(V, PointerType::getUnqual(B.getInt8Ty()), "cstr");
}

/// EmitMemCpy - Emit a call to the memcpy function to the builder.  This always
/// expects that the size has type 'intptr_t' and Dst/Src are pointers.
Value *LibCallOptimization::EmitMemCpy(Value *Dst, Value *Src, Value *Len,
                                       unsigned Align, IRBuilder<> &B) {
  return B.CreateMemCpy(CastToCStr(Dst, B), CastToCStr(Src, B), Len, Align,
                        false);
}

//===----------------------------------------------------------------------===//
// Miscellaneous LibCall Optimizations
//===----------------------------------------------------------------------===//

namespace {
//===---------------------------------------===//
// '_d_arraysetlengthT'/'_d_arraysetlengthiT' Optimizations

/// ArraySetLengthOpt - remove libcall for arr.length = N if N <= arr.length
struct LLVM_LIBRARY_VISIBILITY ArraySetLengthOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify we have a reasonable prototype for _d_arraysetlength[i]T
    const FunctionType *FT = Callee->getFunctionType();
    if (Callee->arg_size() != 4 || !isa<PointerType>(FT->getReturnType()) ||
        !isa<IntegerType>(FT->getParamType(1)) ||
        FT->getParamType(1) != FT->getParamType(2) ||
        FT->getParamType(3) != FT->getReturnType())
      return 0;

    // Whether or not this allocates is irrelevant if the result isn't used.
    // Just delete if that's the case.
    if (CI->use_empty())
      return CI;

    Value *NewLen = CI->getOperand(1);
    if (Constant *NewCst = dyn_cast<Constant>(NewLen)) {
      Value *Data = CI->getOperand(3);

      // For now, we just catch the simplest of cases.
      //
      // TODO: Implement a more general way to compare old and new
      //       lengths, to catch cases like "arr.length = arr.length - 1;"
      //       (But beware of unsigned overflow! For example, we can't
      //       safely transform that example if arr.length may be 0)

      // Setting length to 0 never reallocates, so replace by data argument
      if (NewCst->isNullValue())
        return Data;

      // If both lengths are constant integers, see if NewLen <= OldLen
      Value *OldLen = CI->getOperand(2);
      if (ConstantInt *OldInt = dyn_cast<ConstantInt>(OldLen))
        if (ConstantInt *NewInt = dyn_cast<ConstantInt>(NewCst))
          if (NewInt->getValue().ule(OldInt->getValue()))
            return Data;
    }
    return 0;
  }
};

/// ArrayCastLenOpt - remove libcall for cast(T[]) arr if it's safe to do so.
struct LLVM_LIBRARY_VISIBILITY ArrayCastLenOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify we have a reasonable prototype for _d_array_cast_len
    const FunctionType *FT = Callee->getFunctionType();
    const Type *RetTy = FT->getReturnType();
    if (Callee->arg_size() != 3 || !isa<IntegerType>(RetTy) ||
        FT->getParamType(0) != RetTy || FT->getParamType(1) != RetTy ||
        FT->getParamType(2) != RetTy)
      return 0;

    Value *OldLen = CI->getOperand(0);
    Value *OldSize = CI->getOperand(1);
    Value *NewSize = CI->getOperand(2);

    // If the old length was zero, always return zero.
    if (Constant *LenCst = dyn_cast<Constant>(OldLen))
      if (LenCst->isNullValue())
        return OldLen;

    // Equal sizes are much faster to check for, so do so now.
    if (OldSize == NewSize)
      return OldLen;

    // If both sizes are constant integers, see if OldSize is a multiple of
    // NewSize
    if (ConstantInt *OldInt = dyn_cast<ConstantInt>(OldSize))
      if (ConstantInt *NewInt = dyn_cast<ConstantInt>(NewSize)) {
        // Don't crash on NewSize == 0, even though it shouldn't happen.
        if (NewInt->isNullValue())
          return 0;

        APInt Quot, Rem;
        APInt::udivrem(OldInt->getValue(), NewInt->getValue(), Quot, Rem);
        if (Rem == 0)
          return B.CreateMul(OldLen, ConstantInt::get(*Context, Quot));
      }
    return 0;
  }
};

/// AllocationOpt - Common optimizations for various GC allocations.
struct LLVM_LIBRARY_VISIBILITY AllocationOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Allocations are never equal to constants, so remove any equality
    // comparisons to constants. (Most importantly comparisons to null at
    // the start of inlined member functions)
    for (CallInst::use_iterator I = CI->use_begin(), E = CI->use_end();
         I != E;) {
      Instruction *User = cast<Instruction>(*I++);

      if (ICmpInst *Cmp = dyn_cast<ICmpInst>(User)) {
        if (!Cmp->isEquality())
          continue;
        Constant *C = 0;
        if ((C = dyn_cast<Constant>(Cmp->getOperand(0))) ||
            (C = dyn_cast<Constant>(Cmp->getOperand(1)))) {
          Value *Result =
              ConstantInt::get(B.getInt1Ty(), !Cmp->isTrueWhenEqual());
          Cmp->replaceAllUsesWith(Result);
          // Don't delete the comparison because there may be an
          // iterator to it. Instead, set the operands to constants
          // and let dead code elimination clean it up later.
          // (It doesn't matter that this changes the value of the
          // icmp because it's not used anymore anyway)
          Cmp->setOperand(0, C);
          Cmp->setOperand(1, C);
          *Changed = true;
        }
      }
    }

    // If it's not used (anymore), pre-emptively GC it.
    if (CI->use_empty())
      return CI;
    return 0;
  }
};

/// ArraySliceCopyOpt - Turn slice copies into llvm.memcpy when safe
struct LLVM_LIBRARY_VISIBILITY ArraySliceCopyOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify we have a reasonable prototype for _d_array_slice_copy
    const FunctionType *FT = Callee->getFunctionType();
    const Type *VoidPtrTy = PointerType::getUnqual(B.getInt8Ty());
    if (Callee->arg_size() != 4 || FT->getReturnType() != B.getVoidTy() ||
        FT->getParamType(0) != VoidPtrTy ||
        !isa<IntegerType>(FT->getParamType(1)) ||
        FT->getParamType(2) != VoidPtrTy ||
        FT->getParamType(3) != FT->getParamType(1))
      return 0;

    Value *Size = CI->getOperand(1);

    // Check the lengths match
    if (CI->getOperand(3) != Size)
      return 0;

    // Assume unknown size unless we have constant size (that fits in an uint)
    unsigned Sz = ~0U;
    if (ConstantInt *Int = dyn_cast<ConstantInt>(Size))
      if (Int->getValue().isIntN(32))
        Sz = Int->getValue().getZExtValue();

    // Check if the pointers may alias
    if (AA->alias(CI->getOperand(0), Sz, CI->getOperand(2), Sz))
      return 0;

    // Equal length and the pointers definitely don't alias, so it's safe to
    // replace the call with memcpy
    return EmitMemCpy(CI->getOperand(0), CI->getOperand(2), Size, 1, B);
  }
};

// TODO: More optimizations! :)

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// SimplifyDRuntimeCalls Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This pass optimizes library functions from the D runtime as used by LDC.
///
class LLVM_LIBRARY_VISIBILITY SimplifyDRuntimeCalls : public FunctionPass {
  StringMap<LibCallOptimization *> Optimizations;

  // Array operations
  ArraySetLengthOpt ArraySetLength;
  ArrayCastLenOpt ArrayCastLen;
  ArraySliceCopyOpt ArraySliceCopy;

  // GC allocations
  AllocationOpt Allocation;

public:
  static char ID; // Pass identification
  SimplifyDRuntimeCalls() : FunctionPass(ID) {}

  void InitOptimizations();
  bool runOnFunction(Function &F);

  bool runOnce(Function &F, const DataLayout *DL, AliasAnalysisPass &AA);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
#if LDC_LLVM_VER >= 307
// The DataLayoutPass is removed.
#else
    AU.addRequired<DataLayoutPass>();
#endif
    AU.addRequired<AliasAnalysisPass>();
  }
};
char SimplifyDRuntimeCalls::ID = 0;
} // end anonymous namespace.

static RegisterPass<SimplifyDRuntimeCalls> X("simplify-drtcalls",
                                             "Simplify calls to D runtime");

// Public interface to the pass.
FunctionPass *createSimplifyDRuntimeCalls() {
  return new SimplifyDRuntimeCalls();
}

/// Optimizations - Populate the Optimizations map with all the optimizations
/// we know.
void SimplifyDRuntimeCalls::InitOptimizations() {
  // Some array-related optimizations
  Optimizations["_d_arraysetlengthT"] = &ArraySetLength;
  Optimizations["_d_arraysetlengthiT"] = &ArraySetLength;
  Optimizations["_d_array_cast_len"] = &ArrayCastLen;
  Optimizations["_d_array_slice_copy"] = &ArraySliceCopy;

  /* Delete calls to runtime functions which aren't needed if their result is
   * unused. That comes down to functions that don't do anything but
   * GC-allocate and initialize some memory.
   * We don't need to do this for functions which are marked 'readnone' or
   * 'readonly', since LLVM doesn't need our help figuring out when those can
   * be deleted.
   * (We can't mark allocating calls as readonly/readnone because they don't
   * return the same pointer every time when called with the same arguments)
   */
  Optimizations["_d_allocmemoryT"] = &Allocation;
  Optimizations["_d_newarrayT"] = &Allocation;
  Optimizations["_d_newarrayiT"] = &Allocation;
  Optimizations["_d_newarrayU"] = &Allocation;
  Optimizations["_d_newarraymT"] = &Allocation;
  Optimizations["_d_newarraymiT"] = &Allocation;
  Optimizations["_d_newarraymvT"] = &Allocation;
  Optimizations["_d_newclass"] = &Allocation;
}

/// runOnFunction - Top level algorithm.
///
bool SimplifyDRuntimeCalls::runOnFunction(Function &F) {
  if (Optimizations.empty())
    InitOptimizations();

#if LDC_LLVM_VER >= 307
  const DataLayout *DL = &F.getParent()->getDataLayout();
#else
  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  const DataLayout *DL = DLP ? &DLP->getDataLayout() : 0;
#endif
  AliasAnalysisPass &AA = getAnalysis<AliasAnalysisPass>();

  // Iterate to catch opportunities opened up by other optimizations,
  // such as calls that are only used as arguments to unused calls:
  // When the second call gets deleted the first call will become unused, but
  // without iteration we wouldn't notice if we inspected the first call
  // before the second one.
  bool EverChanged = false;
  bool Changed;
  do {
    Changed = runOnce(F, DL, AA);
    EverChanged |= Changed;
  } while (Changed);

  return EverChanged;
}

bool SimplifyDRuntimeCalls::runOnce(Function &F, const DataLayout *DL,
                                    AliasAnalysisPass &AAP) {
  IRBuilder<> Builder(F.getContext());

  bool Changed = false;
  for (auto &BB : F) {
    for (auto I = BB.begin(); I != BB.end();) {
      // Ignore non-calls.
      CallInst *CI = dyn_cast<CallInst>(I++);
      if (!CI)
        continue;

      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CI->getCalledFunction();
      if (Callee == nullptr || !Callee->isDeclaration() ||
          !Callee->hasExternalLinkage())
        continue;

      // Ignore unknown calls.
      auto OMI = Optimizations.find(Callee->getName());
      if (OMI == Optimizations.end())
        continue;

      DEBUG(errs() << "SimplifyDRuntimeCalls inspecting: " << *CI);

      // Set the builder to the instruction after the call.
      Builder.SetInsertPoint(&BB, I);

// Try to optimize this call.
#if LDC_LLVM_VER >= 308
      AliasAnalysis &AA = AAP.getAAResults();
#else
      AliasAnalysis &AA = AAP;
#endif
      Value *Result = OMI->second->OptimizeCall(CI, Changed, DL, AA, Builder);
      if (Result == 0)
        continue;

      DEBUG(errs() << "SimplifyDRuntimeCalls simplified: " << *CI;
            errs() << "  into: " << *Result << "\n");

      // Something changed!
      Changed = true;

      if (Result == CI) {
        assert(CI->use_empty());
        ++NumDeleted;
#if LDC_LLVM_VER < 308
        AA.deleteValue(CI);
#endif
      } else {
        ++NumSimplified;
#if LDC_LLVM_VER < 308
        AA.replaceWithNewValue(CI, Result);
#endif

        if (!CI->use_empty())
          CI->replaceAllUsesWith(Result);

        if (!Result->hasName())
          Result->takeName(CI);
      }

      // Inspect the instruction after the call (which was potentially just
      // added) next.
      I = CI;
      ++I;

      CI->eraseFromParent();
    }
  }
  return Changed;
}
