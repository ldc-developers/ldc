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
#include "llvm/Intrinsics.h"
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
// Helper functions
//===----------------------------------------------------------------------===//

void EmitMemSet(IRBuilder<>& B, Value* Dst, Value* Val, Value* Len) {
    Dst = B.CreateBitCast(Dst, PointerType::getUnqual(Type::Int8Ty));
    
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    const Type* Tys[1];
    Tys[0] = Len->getType();
    Value *MemSet = Intrinsic::getDeclaration(M, Intrinsic::memset, Tys, 1);
    Value *Align = ConstantInt::get(Type::Int32Ty, 1);
    
    B.CreateCall4(MemSet, Dst, Val, Len, Align);
}

static void EmitMemZero(IRBuilder<>& B, Value* Dst, Value* Len) {
    EmitMemSet(B, Dst, ConstantInt::get(Type::Int8Ty, 0), Len);
}


//===----------------------------------------------------------------------===//
// Helpers for specific types of GC calls.
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
        virtual AllocaInst* promote(CallSite CS, IRBuilder<>& B, const Analysis& A) {
            NumGcToStack++;
            
            Instruction* Begin = CS.getCaller()->getEntryBlock().begin();
            return new AllocaInst(Ty, ".nongc_mem", Begin); // FIXME: align?
        }
        
        FunctionInfo(unsigned typeInfoArgNr, bool safeToDelete)
        : TypeInfoArgNr(typeInfoArgNr), SafeToDelete(safeToDelete) {}
    };
    
    class ArrayFI : public FunctionInfo {
        Value* arrSize;
        int ArrSizeArgNr;
        bool Initialized;
        
    public:
        ArrayFI(unsigned tiArgNr, bool safeToDelete, bool initialized,
                unsigned arrSizeArgNr)
        : FunctionInfo(tiArgNr, safeToDelete),
          ArrSizeArgNr(arrSizeArgNr),
          Initialized(initialized)
        {}
        
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
        
        virtual AllocaInst* promote(CallSite CS, IRBuilder<>& B, const Analysis& A) {
            IRBuilder<> Builder = B;
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
            Value* count = Builder.CreateIntCast(arrSize, Type::Int32Ty, false);
            AllocaInst* alloca = Builder.CreateAlloca(Ty, count, ".nongc_mem"); // FIXME: align?
            
            if (Initialized) {
                // For now, only zero-init is supported.
                uint64_t size = A.TD.getTypeStoreSize(Ty);
                Value* TypeSize = ConstantInt::get(arrSize->getType(), size);
                // Use the original B to put initialization at the
                // allocation site.
                Value* Size = B.CreateMul(TypeSize, arrSize);
                EmitMemZero(B, alloca, Size);
            }
            
            return alloca;
        }
    };
    
    // FunctionInfo for _d_allocclass
    class AllocClassFI : public FunctionInfo {
        public:
        virtual bool analyze(CallSite CS, const Analysis& A) {
            // This call contains no TypeInfo parameter, so don't call the
            // base class implementation here...
            if (CS.arg_size() != 1)
                return false;
            Value* arg = CS.getArgument(0)->stripPointerCasts();
            GlobalVariable* ClassInfo = dyn_cast<GlobalVariable>(arg);
            if (!ClassInfo)
                return false;
            
            std::string metaname = CD_PREFIX;
            metaname.append(ClassInfo->getNameStart(), ClassInfo->getNameEnd());
            
            GlobalVariable* global = A.M.getGlobalVariable(metaname);
            if (!global || !global->hasInitializer())
                return false;
            
            MDNode* node = dyn_cast<MDNode>(global->getInitializer());
            if (!node || MD_GetNumElements(node) != CD_NumFields)
                return false;
            
            // Inserting destructor calls is not implemented yet, so classes
            // with destructors are ignored for now.
            Constant* hasDestructor = dyn_cast<Constant>(MD_GetElement(node, CD_Finalize));
            // We can't stack-allocate if the class has a custom deallocator
            // (Custom allocators don't get turned into this runtime call, so
            // those can be ignored)
            Constant* hasCustomDelete = dyn_cast<Constant>(MD_GetElement(node, CD_CustomDelete));
            if (hasDestructor == NULL || hasCustomDelete == NULL)
                return false;
            
            if (ConstantExpr::getOr(hasDestructor, hasCustomDelete)
                    != ConstantInt::getFalse())
                return false;
            
            Ty = MD_GetElement(node, CD_BodyType)->getType();
            return true;
        }
        
        // The default promote() should be fine.
        
        AllocClassFI() : FunctionInfo(-1, true) {}
    };
}


//===----------------------------------------------------------------------===//
// GarbageCollect2Stack Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
    /// This pass replaces GC calls with alloca's
    ///
    class VISIBILITY_HIDDEN GarbageCollect2Stack : public FunctionPass {
        StringMap<FunctionInfo*> KnownFunctions;
        Module* M;
        
        FunctionInfo AllocMemoryT;
        ArrayFI NewArrayVT;
        ArrayFI NewArrayT;
        AllocClassFI AllocClass;
        
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
  NewArrayVT(0, true, false, 1),
  NewArrayT(0, true, true, 1)
{
    KnownFunctions["_d_allocmemoryT"] = &AllocMemoryT;
    KnownFunctions["_d_newarrayvT"] = &NewArrayVT;
    KnownFunctions["_d_newarrayT"] = &NewArrayT;
    KnownFunctions["_d_allocclass"] = &AllocClass;
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
            
            IRBuilder<> Builder(BB, Inst);
            Value* newVal = info->promote(CS, Builder, A);
            
            DEBUG(DOUT << "Promoted to: " << *newVal);
            
            // Make sure the type is the same as it was before, and replace all
            // uses of the runtime call with the alloca.
            if (newVal->getType() != Inst->getType())
                newVal = Builder.CreateBitCast(newVal, Inst->getType());
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
    
    if (MD_GetNumElements(node) != TD_NumFields)
        return NULL;
    if (TD_Confirm >= 0 && (!MD_GetElement(node, TD_Confirm) ||
            MD_GetElement(node, TD_Confirm)->stripPointerCasts() != ti_global))
        return NULL;
    
    return MD_GetElement(node, TD_Type)->getType();
}


#endif //USE_METADATA
