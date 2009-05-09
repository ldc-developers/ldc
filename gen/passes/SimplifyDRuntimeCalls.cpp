//===- SimplifyDRuntimeCalls - Optimize calls to the D runtime library ----===//
//
//                             The LLVM D Compiler
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of runtime calls simplified");
STATISTIC(NumDeleted, "Number of runtime calls deleted");

//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
    class VISIBILITY_HIDDEN LibCallOptimization {
    protected:
        Function *Caller;
        const TargetData *TD;
    public:
        LibCallOptimization() { }
        virtual ~LibCallOptimization() {}
        
        /// CallOptimizer - This pure virtual method is implemented by base classes to
        /// do various optimizations.  If this returns null then no transformation was
        /// performed.  If it returns CI, then it transformed the call and CI is to be
        /// deleted.  If it returns something else, replace CI with the new value and
        /// delete CI.
        virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B)=0;
        
        Value *OptimizeCall(CallInst *CI, const TargetData &TD, IRBuilder<> &B) {
            Caller = CI->getParent()->getParent();
            this->TD = &TD;
            return CallOptimizer(CI->getCalledFunction(), CI, B);
        }
    };
} // End anonymous namespace.


//===----------------------------------------------------------------------===//
// Miscellaneous LibCall Optimizations
//===----------------------------------------------------------------------===//

namespace {
//===---------------------------------------===//
// '_d_arraysetlengthT'/'_d_arraysetlengthiT' Optimizations

/// ArraySetLengthOpt - remove libcall for arr.length = N if N <= arr.length
struct VISIBILITY_HIDDEN ArraySetLengthOpt : public LibCallOptimization {
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
        
        Value* NewLen = CI->getOperand(2);
        if (Constant* NewCst = dyn_cast<Constant>(NewLen)) {
            Value* Data = CI->getOperand(4);
            
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
            Value* OldLen = CI->getOperand(3);
            if (ConstantInt* OldInt = dyn_cast<ConstantInt>(OldLen))
                if (ConstantInt* NewInt = dyn_cast<ConstantInt>(NewCst))
                    if (NewInt->getValue().ule(OldInt->getValue()))
                        return Data;
        }
        return 0;
    }
};

/// ArrayCastLenOpt - remove libcall for cast(T[]) arr if it's safe to do so.
struct VISIBILITY_HIDDEN ArrayCastLenOpt : public LibCallOptimization {
    virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
        // Verify we have a reasonable prototype for _d_array_cast_len
        const FunctionType *FT = Callee->getFunctionType();
        const Type* RetTy = FT->getReturnType();
        if (Callee->arg_size() != 3 || !isa<IntegerType>(RetTy) ||
            FT->getParamType(1) != RetTy || FT->getParamType(2) != RetTy)
          return 0;
        
        Value* OldLen = CI->getOperand(1);
        Value* OldSize = CI->getOperand(2);
        Value* NewSize = CI->getOperand(3);
        
        // If the old length was zero, always return zero.
        if (Constant* LenCst = dyn_cast<Constant>(OldLen))
            if (LenCst->isNullValue())
                return OldLen;
        
        // Equal sizes are much faster to check for, so do so now.
        if (OldSize == NewSize)
            return OldLen;
        
        // If both sizes are constant integers, see if OldSize is a multiple of NewSize
        if (ConstantInt* OldInt = dyn_cast<ConstantInt>(OldSize))
            if (ConstantInt* NewInt = dyn_cast<ConstantInt>(NewSize)) {
                // Don't crash on NewSize == 0, even though it shouldn't happen.
                if (NewInt->isNullValue())
                    return 0;
                
                APInt Quot, Rem;
                APInt::udivrem(OldInt->getValue(), NewInt->getValue(), Quot, Rem);
                if (Rem == 0)
                    return B.CreateMul(OldLen, ConstantInt::get(Quot));
            }
        return 0;
    }
};

/// DeleteUnusedOpt - remove libcall if the return value is unused.
struct VISIBILITY_HIDDEN DeleteUnusedOpt : public LibCallOptimization {
    virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
        if (CI->use_empty())
            return CI;
        return 0;
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
    class VISIBILITY_HIDDEN SimplifyDRuntimeCalls : public FunctionPass {
        StringMap<LibCallOptimization*> Optimizations;
        
        // Array operations
        ArraySetLengthOpt ArraySetLength;
        ArrayCastLenOpt ArrayCastLen;
        
        // GC allocations
        DeleteUnusedOpt DeleteUnused;
        
        public:
        static char ID; // Pass identification
        SimplifyDRuntimeCalls() : FunctionPass(&ID) {}
        
        void InitOptimizations();
        bool runOnFunction(Function &F);
        
        bool runOnce(Function &F, const TargetData& TD);
            
        virtual void getAnalysisUsage(AnalysisUsage &AU) const {
          AU.addRequired<TargetData>();
        }
    };
    char SimplifyDRuntimeCalls::ID = 0;
} // end anonymous namespace.

static RegisterPass<SimplifyDRuntimeCalls>
X("simplify-drtcalls", "Simplify calls to D runtime");

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
    
    /* Delete calls to runtime functions which aren't needed if their result is
     * unused. That comes down to functions that don't do anything but
     * GC-allocate and initialize some memory.
     * We don't need to do this for functions which are marked 'readnone' or
     * 'readonly', since LLVM doesn't need our help figuring out when those can
     * be deleted.
     * (We can't mark allocating calls as readonly/readnone because they don't
     * return the same pointer every time when called with the same arguments)
     */
    Optimizations["_d_allocmemoryT"] = &DeleteUnused;
    Optimizations["_d_newarrayT"] = &DeleteUnused;
    Optimizations["_d_newarrayiT"] = &DeleteUnused;
    Optimizations["_d_newarrayvT"] = &DeleteUnused;
    Optimizations["_d_newarraymT"] = &DeleteUnused;
    Optimizations["_d_newarraymiT"] = &DeleteUnused;
    Optimizations["_d_newarraymvT"] = &DeleteUnused;
    Optimizations["_d_allocclass"] = &DeleteUnused;
}


/// runOnFunction - Top level algorithm.
///
bool SimplifyDRuntimeCalls::runOnFunction(Function &F) {
    if (Optimizations.empty())
        InitOptimizations();
    
    const TargetData &TD = getAnalysis<TargetData>();
    
    // Iterate to catch opportunities opened up by other optimizations,
    // such as calls that are only used as arguments to unused calls:
    // When the second call gets deleted the first call will become unused, but
    // without iteration we wouldn't notice if we inspected the first call
    // before the second one.
    bool EverChanged = false;
    bool Changed;
    do {
        Changed = runOnce(F, TD);
        EverChanged |= Changed;
    } while (Changed);
    
    return EverChanged;
}

bool SimplifyDRuntimeCalls::runOnce(Function &F, const TargetData& TD) {
    IRBuilder<> Builder;
    
    bool Changed = false;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
            // Ignore non-calls.
            CallInst *CI = dyn_cast<CallInst>(I++);
            if (!CI) continue;
            
            // Ignore indirect calls and calls to non-external functions.
            Function *Callee = CI->getCalledFunction();
            if (Callee == 0 || !Callee->isDeclaration() ||
                    !(Callee->hasExternalLinkage() || Callee->hasDLLImportLinkage()))
                continue;
            
            DEBUG(DOUT << "SimplifyDRuntimeCalls inspecting: " << *CI);
            
            // Ignore unknown calls.
            const char *CalleeName = Callee->getNameStart();
            StringMap<LibCallOptimization*>::iterator OMI =
                Optimizations.find(CalleeName, CalleeName+Callee->getNameLen());
            if (OMI == Optimizations.end()) continue;
            
            // Set the builder to the instruction after the call.
            Builder.SetInsertPoint(BB, I);
            
            // Try to optimize this call.
            Value *Result = OMI->second->OptimizeCall(CI, TD, Builder);
            if (Result == 0) continue;
            
            DEBUG(DOUT << "SimplifyDRuntimeCalls simplified: " << *CI;
                  DOUT << "  into: " << *Result << "\n");
            
            // Something changed!
            Changed = true;
            
            if (Result == CI) {
                assert(CI->use_empty());
                ++NumDeleted;
            } else {
                ++NumSimplified;
                
                if (!CI->use_empty())
                    CI->replaceAllUsesWith(Result);
                
                if (!Result->hasName())
                    Result->takeName(CI);
            }
            
            // Inspect the instruction after the call (which was potentially just
            // added) next.
            I = CI; ++I;
            
            CI->eraseFromParent();
        }
    }
    return Changed;
}
