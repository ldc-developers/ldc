//===- GarbageCollect2Stack - Optimize calls to the D garbage collector ---===//
//
//                             The LLVM D Compiler
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file attempts to turn allocations on the garbage-collected heap into
// stack allocations.
//
//===----------------------------------------------------------------------===//

#include "gen/metadata.h"

// This pass doesn't work without metadata, so #ifdef it out entirely if the
// LLVM version in use doesn't support it.
#ifdef USE_METADATA


#define DEBUG_TYPE "dgc2stack"

#include "Passes.h"

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

STATISTIC(NumGcToStack, "Number of calls promoted to constant-size allocas");
STATISTIC(NumToDynSize, "Number of calls promoted to dynamically-sized allocas");
STATISTIC(NumDeleted, "Number of GC calls deleted because the return value was unused");

//===----------------------------------------------------------------------===//
// GarbageCollect2Stack Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
    struct Analysis {
        TargetData& TD;
        const Module& M;
        
        const Type* getTypeFor(Value* typeinfo) const;
    };
    
    class FunctionInfo {
    protected:
        const Type* Ty;
        
    public:
        unsigned TypeInfoArgNr;
        bool SafeToDelete;
        
        // Analyze the current call, filling in some fields. Returns true if
        // this is an allocation we can stack-allocate.
        virtual bool analyze(CallSite CS, const Analysis& A) {
            Value* TypeInfo = CS.getArgument(TypeInfoArgNr);
            Ty = A.getTypeFor(TypeInfo);
            return (Ty != NULL);
        }
        
        // Returns the alloca to replace this call.
        // It will always be inserted before the call.
        virtual AllocaInst* promote(CallSite CS) {
            NumGcToStack++;
            
            Instruction* Begin = CS.getCaller()->getEntryBlock().begin();
            return new AllocaInst(Ty, ".nongc_mem", Begin);
        }
        
        FunctionInfo(unsigned typeInfoArgNr, bool safeToDelete)
        : TypeInfoArgNr(typeInfoArgNr), SafeToDelete(safeToDelete) {}
    };
    
    class ArrayFI : public FunctionInfo {
        Value* arrSize;
        int ArrSizeArgNr;
        
    public:
        ArrayFI(unsigned tiArgNr, bool safeToDelete, unsigned arrSizeArgNr)
        : FunctionInfo(tiArgNr, safeToDelete), ArrSizeArgNr(arrSizeArgNr) {}
        
        virtual bool analyze(CallSite CS, const Analysis& A) {
            if (!FunctionInfo::analyze(CS, A))
                return false;
            
            arrSize = CS.getArgument(ArrSizeArgNr);
            const IntegerType* SizeType =
                dyn_cast<IntegerType>(arrSize->getType());
            if (!SizeType)
                return false;
            unsigned bits = SizeType->getBitWidth();
            if (bits > 32) {
                // The array size of an alloca must be an i32, so make sure
                // the conversion is safe.
                APInt Mask = APInt::getHighBitsSet(bits, bits - 32);
                APInt KnownZero(bits, 0), KnownOne(bits, 0);
                ComputeMaskedBits(arrSize, Mask, KnownZero, KnownOne, &A.TD);
                if ((KnownZero & Mask) != Mask)
                    return false;
            }
            // Extract the element type from the array type.
            const StructType* ArrTy = dyn_cast<StructType>(Ty);
            assert(ArrTy && "Dynamic array type not a struct?");
            assert(isa<IntegerType>(ArrTy->getElementType(0)));
            const PointerType* PtrTy =
                cast<PointerType>(ArrTy->getElementType(1));
            Ty = PtrTy->getElementType();
            return true;
        }
        
        virtual AllocaInst* promote(CallSite CS) {
            Instruction* I = CS.getInstruction();
            IRBuilder<> Builder(I->getParent(), I);
            
            // If the allocation is of constant size it's best to put it in the
            // entry block, so do so if we're not already there.
            // For dynamically-sized allocations it's best to avoid the overhead
            // of allocating them if possible, so leave those where they are.
            // While we're at it, update statistics too.
            if (isa<Constant>(arrSize)) {
                BasicBlock& Entry = CS.getCaller()->getEntryBlock();
                if (Builder.GetInsertBlock() != &Entry)
                    Builder.SetInsertPoint(&Entry, Entry.begin());
                NumGcToStack++;
            } else {
                NumToDynSize++;
            }
            
            // Convert array size to 32 bits if necessary
            arrSize = Builder.CreateIntCast(arrSize, Type::Int32Ty, false);
            
            return Builder.CreateAlloca(Ty, arrSize, ".nongc_mem");
        }
    };
    
    /// This pass replaces GC calls with alloca's
    ///
    class VISIBILITY_HIDDEN GarbageCollect2Stack : public FunctionPass {
        StringMap<FunctionInfo*> KnownFunctions;
        Module* M;
        
        FunctionInfo AllocMemoryT;
        ArrayFI NewArrayVT;
        
    public:
        static char ID; // Pass identification
        GarbageCollect2Stack();
        
        bool doInitialization(Module &M) {
            this->M = &M;
        }
        
        bool runOnFunction(Function &F);
        
        virtual void getAnalysisUsage(AnalysisUsage &AU) const {
          AU.addRequired<TargetData>();
          AU.addRequired<LoopInfo>();
        }
    };
    char GarbageCollect2Stack::ID = 0;
} // end anonymous namespace.

static RegisterPass<GarbageCollect2Stack>
X("dgc2stack", "Promote (GC'ed) heap allocations to stack");

// Public interface to the pass.
FunctionPass *createGarbageCollect2Stack() {
  return new GarbageCollect2Stack(); 
}

GarbageCollect2Stack::GarbageCollect2Stack()
: FunctionPass(&ID),
  AllocMemoryT(0, true),
  NewArrayVT(0, true, 1)
{
    KnownFunctions["_d_allocmemoryT"] = &AllocMemoryT;
    KnownFunctions["_d_newarrayvT"] = &NewArrayVT;
}

static void RemoveCall(Instruction* Inst) {
    if (InvokeInst* Invoke = dyn_cast<InvokeInst>(Inst)) {
        // If this was an invoke instruction, we need to do some extra
        // work to preserve the control flow.
        
        // First notify the exception landing pad block that we won't be
        // going there anymore.
        Invoke->getUnwindDest()->removePredecessor(Invoke->getParent());
        // Create a branch to the "normal" destination.
        BranchInst::Create(Invoke->getNormalDest(), Invoke->getParent());
    }
    // Remove the runtime call.
    Inst->eraseFromParent();
}

/// runOnFunction - Top level algorithm.
///
bool GarbageCollect2Stack::runOnFunction(Function &F) {
    DEBUG(DOUT << "Running -dgc2stack on function " << F.getName() << '\n');
    
    TargetData &TD = getAnalysis<TargetData>();
    const LoopInfo &LI = getAnalysis<LoopInfo>();
    
    Analysis A = { TD, *M };
    
    BasicBlock& Entry = F.getEntryBlock();
    
    IRBuilder<> AllocaBuilder(&Entry, Entry.begin());
    
    bool Changed = false;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
        // We don't yet have sufficient analysis to properly determine if
        // allocations will be unreferenced when the loop returns to their
        // allocation point, so we're playing it safe by ignoring allocations
        // in loops.
        // TODO: Analyze loops too...
        if (LI.getLoopFor(BB)) {
            continue;
        }
        
        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
            // Ignore non-calls.
            Instruction* Inst = I++;
            CallSite CS = CallSite::get(Inst);
            if (!CS.getInstruction())
                continue;
            
            // Ignore indirect calls and calls to non-external functions.
            Function *Callee = CS.getCalledFunction();
            if (Callee == 0 || !Callee->isDeclaration() ||
                    !(Callee->hasExternalLinkage() || Callee->hasDLLImportLinkage()))
                continue;
            
            // Ignore unknown calls.
            const char *CalleeName = Callee->getNameStart();
            StringMap<FunctionInfo*>::iterator OMI =
                KnownFunctions.find(CalleeName, CalleeName+Callee->getNameLen());
            if (OMI == KnownFunctions.end()) continue;
            
            assert(isa<PointerType>(Inst->getType())
                && "GC function doesn't return a pointer?");
            
            FunctionInfo* info = OMI->getValue();
            
            if (Inst->use_empty() && info->SafeToDelete) {
                Changed = true;
                NumDeleted++;
                RemoveCall(Inst);
                continue;
            }
            
            DEBUG(DOUT << "GarbageCollect2Stack inspecting: " << *Inst);
            
            if (!info->analyze(CS, A) || PointerMayBeCaptured(Inst, true))
                continue;
            
            // Let's alloca this!
            Changed = true;
            
            Value* newVal = info->promote(CS);
            
            // Make sure the type is the same as it was before, and replace all
            // uses of the runtime call with the alloca.
            if (newVal->getType() != Inst->getType())
                newVal = new BitCastInst(newVal, Inst->getType(), "", Inst);
            Inst->replaceAllUsesWith(newVal);
            
            RemoveCall(Inst);
        }
    }
    
    return Changed;
}

const Type* Analysis::getTypeFor(Value* typeinfo) const {
    GlobalVariable* ti_global = dyn_cast<GlobalVariable>(typeinfo->stripPointerCasts());
    if (!ti_global)
        return NULL;
    
    std::string metaname = TD_PREFIX;
    metaname.append(ti_global->getNameStart(), ti_global->getNameEnd());
    
    GlobalVariable* global = M.getGlobalVariable(metaname);
    if (!global || !global->hasInitializer())
        return NULL;
    
    MDNode* node = dyn_cast<MDNode>(global->getInitializer());
    if (!node)
        return NULL;
    
    if (node->getNumOperands() != TD_NumFields ||
            node->getOperand(TD_Confirm)->stripPointerCasts() != ti_global)
        return NULL;
    
    return node->getOperand(TD_Type)->getType();
}


#endif //USE_METADATA
