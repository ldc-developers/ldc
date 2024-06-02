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

#include "gen/passes/Passes.h"
#include "gen/passes/SimplifyDRuntimeCalls.h"
#include "gen/tollvm.h"
#include "gen/runtime.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumSimplified, "Number of runtime calls simplified");
STATISTIC(NumDeleted, "Number of runtime calls deleted");

Value *LibCallOptimization::OptimizeCall(CallInst *CI, bool &Changed, const DataLayout *DL,
                    AliasAnalysis &AA, IRBuilder<> &B) {
  Caller = CI->getParent()->getParent();
  this->Changed = &Changed;
  this->DL = DL;
  this->AA = &AA;
  if (CI->getCalledFunction()) {
    Context = &CI->getCalledFunction()->getContext();
  }
  return CallOptimizer(CI->getCalledFunction(), CI, B);
}

/// EmitMemCpy - Emit a call to the memcpy function to the builder.  This always
/// expects that the size has type 'intptr_t' and Dst/Src are pointers.
Value *LibCallOptimization::EmitMemCpy(Value *Dst, Value *Src, Value *Len,
                                       unsigned Align, IRBuilder<> &B) {
  auto A = llvm::MaybeAlign(Align);
  return B.CreateMemCpy(Dst, A, Src, A, Len, false);
}

//===----------------------------------------------------------------------===//
// Miscellaneous LibCall Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// '_d_arraysetlengthT'/'_d_arraysetlengthiT' Optimizations

Value *ArraySetLengthOpt::CallOptimizer(Function *Callee, CallInst *CI,
                     IRBuilder<> &B) {
  // Verify we have a reasonable prototype for _d_arraysetlength[i]T
  const FunctionType *FT = Callee->getFunctionType();
  if (Callee->arg_size() != 4 || !isa<PointerType>(FT->getReturnType()) ||
      !isa<IntegerType>(FT->getParamType(1)) ||
      FT->getParamType(1) != FT->getParamType(2) ||
      FT->getParamType(3) != FT->getReturnType()) {
    return nullptr;
  }

  // Whether or not this allocates is irrelevant if the result isn't used.
  // Just delete if that's the case.
  if (CI->use_empty()) {
    return CI;
  }

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
    if (NewCst->isNullValue()) {
      return Data;
    }

    // If both lengths are constant integers, see if NewLen <= OldLen
    Value *OldLen = CI->getOperand(2);
    if (ConstantInt *OldInt = dyn_cast<ConstantInt>(OldLen)) {
      if (ConstantInt *NewInt = dyn_cast<ConstantInt>(NewCst)) {
        if (NewInt->getValue().ule(OldInt->getValue())) {
          return Data;
        }
      }
    }
  }
  return nullptr;
}

/// AllocationOpt - Common optimizations for various GC allocations.
Value *AllocationOpt::CallOptimizer(Function *Callee, CallInst *CI,
                     IRBuilder<> &B) {
  // Allocations are never equal to constants, so remove any equality
  // comparisons to constants. (Most importantly comparisons to null at
  // the start of inlined member functions)
  for (CallInst::use_iterator I = CI->use_begin(), E = CI->use_end();
       I != E;) {
    Instruction *User = cast<Instruction>(*I++);

    if (ICmpInst *Cmp = dyn_cast<ICmpInst>(User)) {
      if (!Cmp->isEquality()) {
        continue;
      }
      Constant *C = nullptr;
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
  if (CI->use_empty()) {
    return CI;
  }
  return nullptr;
}


// This module will also be used in jit runtime
// copy these function here to avoid dependencies on rest of compiler
LLIntegerType *DtoSize_t(llvm::LLVMContext &context,
                         const llvm::DataLayout &DL) {
  // the type of size_t does not change once set
  static LLIntegerType *t = nullptr;
  if (t == nullptr) {
    const auto ptrsize = DL.getPointerSize();
    if (ptrsize == 8) {
      t = LLType::getInt64Ty(context);
    } else if (ptrsize == 4) {
      t = LLType::getInt32Ty(context);
    } else if (ptrsize == 2) {
      t = LLType::getInt16Ty(context);
    } else {
      llvm_unreachable("Unsupported size_t width");
    }
  }
  return t;
}

llvm::ConstantInt *DtoConstSize_t(llvm::LLVMContext &context,
                                  const llvm::DataLayout &DL, uint64_t i) {
  return LLConstantInt::get(DtoSize_t(context, DL), i, false);
}

/// ArraySliceCopyOpt - Turn slice copies into llvm.memcpy when safe
Value *ArraySliceCopyOpt::CallOptimizer(Function *Callee, CallInst *CI,
                     IRBuilder<> &B) {
  // Verify we have a reasonable prototype for _d_array_slice_copy
  const FunctionType *FT = Callee->getFunctionType();
  const llvm::Type *VoidPtrTy = PointerType::getUnqual(B.getInt8Ty());
  if (Callee->arg_size() != 5 || FT->getReturnType() != B.getVoidTy() ||
      FT->getParamType(0) != VoidPtrTy ||
      !isa<IntegerType>(FT->getParamType(1)) ||
      FT->getParamType(2) != VoidPtrTy ||
      FT->getParamType(3) != FT->getParamType(1) ||
      FT->getParamType(4) != FT->getParamType(1)) {
    return nullptr;
  }

  Value *DstLength = CI->getOperand(1);

  // Check the lengths match
  if (CI->getOperand(3) != DstLength) {
    return nullptr;
  }

  const auto ElemSz = llvm::cast<ConstantInt>(CI->getOperand(4));

  // Assume unknown size unless we have constant length
  std::uint64_t Sz = llvm::MemoryLocation::UnknownSize;
  if (ConstantInt *Int = dyn_cast<ConstantInt>(DstLength)) {
    Sz = (Int->getValue() * ElemSz->getValue()).getZExtValue();
  }

  // Check if the pointers may alias
  if (AA->alias(CI->getOperand(0), Sz, CI->getOperand(2), Sz)) {
    return nullptr;
  }

  // Equal length and the pointers definitely don't alias, so it's safe to
  // replace the call with memcpy
  auto size = Sz != llvm::MemoryLocation::UnknownSize
                  ? DtoConstSize_t(Callee->getContext(),
                                   Callee->getParent()->getDataLayout(), Sz)
                  : B.CreateMul(DstLength, ElemSz);
  return EmitMemCpy(CI->getOperand(0), CI->getOperand(2), size, 1, B);
}


// TODO: More optimizations! :)


//===----------------------------------------------------------------------===//
// SimplifyDRuntimeCalls Pass Implementation
//===----------------------------------------------------------------------===//



class LLVM_LIBRARY_VISIBILITY SimplifyDRuntimeCallsLegacyPass : public FunctionPass {
  SimplifyDRuntimeCalls pass;

public:
  static char ID; // Pass identification
  SimplifyDRuntimeCallsLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    auto getAA = [&]() -> AAResults& {
      return getAnalysis<AAResultsWrapperPass>().getAAResults();
    };

    return pass.run(F, getAA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
  }
};
char SimplifyDRuntimeCallsLegacyPass::ID = 0;

static RegisterPass<SimplifyDRuntimeCallsLegacyPass>
    X("simplify-drtcalls","Simplify calls to D runtime");

// Public interface to the pass.
FunctionPass *createSimplifyDRuntimeCalls() {
  return new SimplifyDRuntimeCallsLegacyPass();
}

/// Optimizations - Populate the Optimizations map with all the optimizations
/// we know.
void SimplifyDRuntimeCalls::InitOptimizations() {
  // Some array-related optimizations
  Optimizations["_d_arraysetlengthT"] = &ArraySetLength;
  Optimizations["_d_arraysetlengthiT"] = &ArraySetLength;
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
  Optimizations["_d_allocclass"] = &Allocation;
}

/// runOnFunction - Top level algorithm.
///
bool SimplifyDRuntimeCalls::run(Function &F,  std::function<AAResults& ()> getAA) {
  if (Optimizations.empty()) {
    InitOptimizations();
  }

  const DataLayout *DL = &F.getParent()->getDataLayout();

  // Iterate to catch opportunities opened up by other optimizations,
  // such as calls that are only used as arguments to unused calls:
  // When the second call gets deleted the first call will become unused, but
  // without iteration we wouldn't notice if we inspected the first call
  // before the second one.
  bool EverChanged = false;
  bool Changed;
  AAResults& AA = getAA();
  do {
    Changed = runOnce(F, DL, AA);
    EverChanged |= Changed;
  } while (Changed);

  return EverChanged;
}

bool SimplifyDRuntimeCalls::runOnce(Function &F, const DataLayout *DL,
                                    AAResults &AA) {
  IRBuilder<> Builder(F.getContext());

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

      // Ignore unknown calls.
      auto OMI = Optimizations.find(Callee->getName());
      if (OMI == Optimizations.end()) {
        continue;
      }

      LLVM_DEBUG(errs() << "SimplifyDRuntimeCalls inspecting: " << *CI);

      // Save the iterator to the call instruction and set the builder to the
      // next instruction.
      auto ciIt = I; // already advanced
      --ciIt;
      Builder.SetInsertPoint(&BB, I);

      // Try to optimize this call.
      Value *Result = OMI->second->OptimizeCall(CI, Changed, DL, AA, Builder);
      if (Result == nullptr) {
        continue;
      }

      LLVM_DEBUG(errs() << "SimplifyDRuntimeCalls simplified: " << *CI;
                 errs() << "  into: " << *Result << "\n");

      // Something changed!
      Changed = true;

      if (Result == CI) {
        assert(CI->use_empty());
        ++NumDeleted;
      } else {
        ++NumSimplified;

        if (!CI->use_empty()) {
          CI->replaceAllUsesWith(Result);
        }

        if (!Result->hasName()) {
          Result->takeName(CI);
        }
      }

      // Inspect the instruction after the call (which was potentially just
      // added) next.
      I = ciIt;
      ++I;

      CI->eraseFromParent();
    }
  }
  return Changed;
}
