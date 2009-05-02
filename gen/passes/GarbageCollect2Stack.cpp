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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

STATISTIC(NumGcToStack, "Number of GC calls promoted to stack allocations");
STATISTIC(NumDeleted, "Number of GC calls deleted because the return value was unused");

//===----------------------------------------------------------------------===//
// GarbageCollect2Stack Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
    struct FunctionInfo {
        unsigned TypeInfoArgNr;
        bool SafeToDelete;
        
        FunctionInfo(unsigned typeInfoArgNr, bool safeToDelete)
        : TypeInfoArgNr(typeInfoArgNr), SafeToDelete(safeToDelete) {}
        
        Value* getArraySize(CallSite CS) {
            return 0;
        }
    };
    
    /// This pass replaces GC calls with alloca's
    ///
    class VISIBILITY_HIDDEN GarbageCollect2Stack : public FunctionPass {
        StringMap<FunctionInfo*> KnownFunctions;
        Module* M;
        
        public:
        static char ID; // Pass identification
        GarbageCollect2Stack() : FunctionPass(&ID) {}
        
        bool doInitialization(Module &M);
        
        bool runOnFunction(Function &F);
        
        virtual void getAnalysisUsage(AnalysisUsage &AU) const {
          AU.addRequired<TargetData>();
          AU.addRequired<LoopInfo>();
        }
        
        private:
        const Type* getTypeFor(Value* typeinfo);
    };
    char GarbageCollect2Stack::ID = 0;
} // end anonymous namespace.

static RegisterPass<GarbageCollect2Stack>
X("dgc2stack", "Promote (GC'ed) heap allocations to stack");

// Public interface to the pass.
FunctionPass *createGarbageCollect2Stack() {
  return new GarbageCollect2Stack(); 
}

bool GarbageCollect2Stack::doInitialization(Module &M) {
    this->M = &M;
    KnownFunctions["_d_allocmemoryT"] = new FunctionInfo(0, true);
}

/// runOnFunction - Top level algorithm.
///
bool GarbageCollect2Stack::runOnFunction(Function &F) {
    DEBUG(DOUT << "Running on function " << F.getName() << '\n');
    
    const TargetData &TD = getAnalysis<TargetData>();
    const LoopInfo &LI = getAnalysis<LoopInfo>();
    
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
            DEBUG(DOUT << "Skipping loop block " << *BB << '\n');
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
                Inst->eraseFromParent();
                continue;
            }
            
            DEBUG(DOUT << "GarbageCollect2Stack inspecting: " << *Inst);
            
            if (PointerMayBeCaptured(Inst, true)) {
                DEBUG(DOUT << ">> is captured :(\n");
                continue;
            }
            DEBUG(DOUT << ">> is not captured :)\n");
            
            Value* TypeInfo = CS.getArgument(info->TypeInfoArgNr);
            const Type* Ty = getTypeFor(TypeInfo);
            if (!Ty) {
                DEBUG(DOUT << ">> Couldn't find valid TypeInfo metadata :(\n");
                continue;
            }
            
            // Let's alloca this!
            Changed = true;
            NumGcToStack++;
            
            Value* arrSize = info->getArraySize(CS);
            Value* newVal = AllocaBuilder.CreateAlloca(Ty, arrSize, ".nongc_mem");
            
            if (newVal->getType() != Inst->getType())
                newVal = AllocaBuilder.CreateBitCast(newVal, Inst->getType());
            
            Inst->replaceAllUsesWith(newVal);
            
            if (InvokeInst* Invoke = dyn_cast<InvokeInst>(Inst)) {
                Invoke->getUnwindDest()->removePredecessor(Invoke->getParent());
                // Create a branch to the "normal" destination.
                BranchInst::Create(Invoke->getNormalDest(), Invoke->getParent());
            }
            Inst->eraseFromParent();
        }
    }
    return Changed;
}

const Type* GarbageCollect2Stack::getTypeFor(Value* typeinfo) {
    GlobalVariable* ti_global = dyn_cast<GlobalVariable>(typeinfo->stripPointerCasts());
    if (!ti_global)
        return NULL;
    
    DEBUG(DOUT << ">> Found typeinfo init\n";
          DOUT << ">> Value: " << *ti_global << "\n";
          DOUT << ">> Name: " << ti_global->getNameStr() << "\n");
    
    std::string metaname = TD_PREFIX;
    metaname.append(ti_global->getNameStart(), ti_global->getNameEnd());
    
    DEBUG(DOUT << ">> Looking for global named " << metaname << "\n");
    
    GlobalVariable* global = M->getGlobalVariable(metaname);
    DEBUG(DOUT << ">> global: " << global->getInitializer() << "\n");
    if (!global || !global->hasInitializer())
        return NULL;
    
    DEBUG(DOUT << ">> Found metadata global\n");
    
    MDNode* node = dyn_cast<MDNode>(global->getInitializer());
    if (!node)
        return NULL;
    
    DEBUG(DOUT << ">> Found metadata node\n");
    
    if (node->getNumOperands() != TD_NumFields ||
            node->getOperand(TD_Confirm)->stripPointerCasts() != ti_global)
        return NULL;
    
    DEBUG(DOUT << ">> Validated metadata node\n");
    
    return node->getOperand(TD_Type)->getType();
}


#endif //USE_METADATA
