//===-- WasmPointersSpill.cpp - Spill pointer onto stack for Wasm ---------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the University of Illinois Open Source
// License. See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to scan vregs and dump potential GC pointers that
// are live across calls (which may trigger GC) back to the stack so they can
// be scanned in Wasm environments.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "wasm-ptrs-spill"

#include "gen/passes/Passes.h"
#include "gen/passes/WasmPointersSpill.h"
#include "gen/llvmhelpers.h"
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
#include "llvm/ADT/PostOrderIterator.h"


using namespace llvm;

class LLVM_LIBRARY_VISIBILITY WasmPointersSpillLegacyPass : public FunctionPass {
  WasmPointersSpill pass;

public:
  static char ID; // Pass identification
  WasmPointersSpillLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    return pass.run(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<AAResultsWrapperPass>();
  }
};
char WasmPointersSpillLegacyPass::ID = 0;

static RegisterPass<WasmPointersSpillLegacyPass>
    X("wasm-ptrs-spill", "Spill ptr vreg to `alloca` for Wasm");

// Public interface to the pass.
FunctionPass *createWasmPointersSpill() {
  return new WasmPointersSpillLegacyPass();
}

static bool TypeContainsPointers(LLType *ty) {
  switch (ty->getTypeID()) {
  case LLType::PointerTyID:
    return true;

  case LLType::ArrayTyID:
    return TypeContainsPointers(ty->getArrayElementType());

  case LLType::StructTyID: {
    unsigned NumElements = ty->getStructNumElements();
    for (unsigned I = 0; I < NumElements; ++I) {
      if (TypeContainsPointers(ty->getStructElementType(I))) {
        return true;
      }
    }
    return false;
  }

  default:
    return false;
  }
}

bool WasmPointersSpill::run(Function &F) {
  IRBuilder<> Builder(F.getContext());
dbgs() << F.getName() << "\n";
  bool Changed = false;

  /*for (auto &BB : F) {
    for (Instruction &I : BB) {
      if (I.getType()->isPointerTy() || TypeContainsPointers(I.getType())) SpillCandidates.push_back(&I);
    }
    }*/

  auto *entryBB = &F.getEntryBlock();

  DenseMap<Instruction *, uint32_t> InstIdx;
  uint64_t NextInstIdx = 0;
  for (auto &BB : F) {
    for (Instruction &I : BB) {
      if (
        !I.getType()->isVoidTy() // we only care about vreg/values
        && !isa<AllocaInst>(I) // the pointer returned by `alloca` isn't in the GC heap
      ) {
        InstIdx[&I] = NextInstIdx++;
      }
      if (NextInstIdx >= (uint64_t(1) << 32)) reportFatalInternalError("More vregs than can fit in 32-bits!");
    }
  }

  size_t NumVRegs = NextInstIdx;

  //SmallSetVector<BasicBlock *, 8> Worklist;
  //DenseMap<BasicBlock *, BitVector> LiveIn;
  //DenseMap<BasicBlock *, BitVector> LiveOut;
  DenseMap<BasicBlock *, BitVector> Defs;
  DenseMap<BasicBlock *, BitVector> Uses;


  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (BasicBlock* BB : make_range(RPOT.begin(), RPOT.end())) {
    //LiveIn[BB].resize(NumVRegs);
    //LiveOut[BB].resize(NumVRegs);
    Defs[BB].resize(NumVRegs);
    Uses[BB].resize(NumVRegs);

    auto &BBUses = Uses[BB];
    auto &BBDefs = Defs[BB];

    BitVector TmpUses(NumVRegs);

    for (Instruction &I : make_range(BB->rbegin(), BB->rend())) {
      for (Value *Op : I.operands()) {
        if (auto *OpInst = dyn_cast<Instruction>(Op)) {
          if (InstIdx.contains(OpInst)) TmpUses.set(InstIdx[OpInst]);
        }
      }

      if (InstIdx.contains(&I)) {
        BBDefs.set(InstIdx[&I]);
        TmpUses.reset(InstIdx[&I]);
      }

      if (isa<CallBase>(&I)) BBUses |= TmpUses;
    }
    //Worklist.insert(BB);
  }

  /*
  while (!Worklist.empty()) {
    auto *BB = Worklist.back();
    Worklist.pop_back();

    BitVector NewLiveOut(NumVRegs);

    for (BasicBlock *Succ : successors(BB)) {
      NewLiveOut |= LiveIn[Succ];
    }

    LiveOut[BB] = NewLiveOut;

    BitVector NewLiveIn = NewLiveOut;
    NewLiveIn |= Uses[BB];
    NewLiveIn.reset(Defs[BB]);

    if (NewLiveIn != LiveIn[BB]) {
      LiveIn[BB] = NewLiveIn;

      for (BasicBlock *Pred : predecessors(BB)) {
        Worklist.insert(Pred);
      }
    }
  }
  */

  BitVector NeedsSpill(NumVRegs);
  for (auto &BB : F) {
    NeedsSpill |= Uses[&BB];
  }


  auto *allocaBB = BasicBlock::Create(F.getContext(), "stackSpillAllocas");
  allocaBB->insertInto(&F, entryBB);

  SmallVector<Instruction *> InstToSpill = to_vector(InstIdx.keys());
  std::sort(InstToSpill.begin(), InstToSpill.end(), [&](Instruction *a, Instruction *b)
    {
        return InstIdx[a] < InstIdx[b];
    }
  );

  for (Instruction *I : InstToSpill) {
    if (!NeedsSpill[InstIdx[I]]) continue;

    auto *ai = new AllocaInst(I->getType(), 0, "stackSpill." + (I->hasName() ? I->getName() : "unnamedVreg"), allocaBB);

    auto *store = new StoreInst(I, ai, true, ai->getAlign(), nullptr);
    store->insertAfter(I);
  }

  UncondBrInst::Create(entryBB, allocaBB);

  return Changed;
}
