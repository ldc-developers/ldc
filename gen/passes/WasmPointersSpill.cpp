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

#include <limits>
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

  case LLType::FixedVectorTyID:
      return TypeContainsPointers(cast<VectorType>(ty)->getElementType());

  default:
    return false;
  }
}

bool WasmPointersSpill::run(Function &F) {
  F.renumberBlocks();

  bool Changed = false;

  // for SetVector determinstic iteration
  SmallSetVector<Instruction *, 0> PotentialPointers;

  {
    SmallPtrSet<Instruction *, 0> WorklistSeen;
    SmallSetVector<Instruction *, 0> Worklist;

    for (auto &BB : F) {
      for (Instruction &I : BB) {
        if (TypeContainsPointers(I.getType())) {
          PotentialPointers.insert(&I);
          if (IntToPtrInst *IntToPtr = dyn_cast<IntToPtrInst>(&I)) {
            if (auto *OpInst = dyn_cast<Instruction>(IntToPtr->getOperand(0))) {
              WorklistSeen.insert(OpInst);
              Worklist.insert(OpInst);
            }
          }
        }
      }
    }

    while (!Worklist.empty()) {
      auto *I = Worklist.back();
      Worklist.pop_back();

      if (I->getType()->isVoidTy()) continue; // we don't care about non-values

      // small integers (and bools) can't hold values large enough to
      // be in the GC heap; assuming we have at least 64K of stack space + data
      if (I->getType()->isIntegerTy()
          && I->getType()->getScalarSizeInBits() <= 16) continue;

      PotentialPointers.insert(I);

      if (isa<LoadInst>(I)) continue; // the operand of the load is unrelated to its result

      // We can safely assume the return of the function is
      // unrelated to any non-pointer operands.
      //
      // However, we can only assume that for D function calls
      // Not intrinsics LLVM might insert.
      if (isa<CallBase>(I) && !isa<IntrinsicInst>(I)) continue;

      for (Use &Op : I->operands()) {
        Value *V = Op.get();

        // If the op is/has a pointer, it's not a concern for
        // hidding values anymore; stop the back-tracking.
        if (TypeContainsPointers(V->getType())) continue;

        if (auto *OpInst = dyn_cast<Instruction>(V)) {
          if (WorklistSeen.insert(OpInst).second) Worklist.insert(OpInst);
        }
      }
    }
  }

  assert(PotentialPointers.size() < std::numeric_limits<unsigned>::max());

  DenseMap<Instruction *, unsigned> InstIdx;
  unsigned NumVRegs = 0;
  for (Instruction *I : PotentialPointers) {
    InstIdx[I] = NumVRegs++;
  }

  SmallSetVector<BasicBlock *, 8> Worklist;
  DenseMap<BasicBlock *, BitVector> LiveIn;
  DenseMap<BasicBlock *, BitVector> LiveInAcrossCall;
  DenseMap<BasicBlock *, BitVector> LiveOut;
  DenseMap<BasicBlock *, BitVector> LiveOutAcrossCall;
  DenseMap<BasicBlock *, BitVector> Defs;
  DenseMap<BasicBlock *, BitVector> Uses;
  DenseMap<BasicBlock *, BitVector> UsesAcrossCall;
  DenseMap<BasicBlock *, BitVector> DefsAfterAllCalls;
  BitVector HasCalls(F.size());

  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (BasicBlock* BB : make_range(RPOT.begin(), RPOT.end())) {
    LiveIn[BB].resize(NumVRegs);
    LiveInAcrossCall[BB].resize(NumVRegs);
    LiveOut[BB].resize(NumVRegs);
    LiveOutAcrossCall[BB].resize(NumVRegs);
    Defs[BB].resize(NumVRegs);
    Uses[BB].resize(NumVRegs);
    UsesAcrossCall[BB].resize(NumVRegs);
    DefsAfterAllCalls[BB].resize(NumVRegs);

    auto &BBUses = Uses[BB];
    auto &BBUsesAcrossCall = UsesAcrossCall[BB];
    auto &BBDefs = Defs[BB];
    auto &BBDefsAfterAllCalls = DefsAfterAllCalls[BB];

    BitVector TmpUses(NumVRegs);
    bool SeenCall = false;

    for (Instruction &I : make_range(BB->rbegin(), BB->rend())) {
      for (Use &Op : I.operands()) {
        Value *V = Op.get();

        if (auto *OpInst = dyn_cast<Instruction>(V)) {
          if (InstIdx.contains(OpInst)) {
            auto Idx = InstIdx[OpInst];
            BBUses.set(Idx);
            TmpUses.set(Idx);
          }
        }
      }

      if (InstIdx.contains(&I)) {
        auto Idx = InstIdx[&I];
        BBDefs.set(Idx);
        if (!SeenCall) BBDefsAfterAllCalls.set(Idx);
        BBUses.reset(Idx);
        TmpUses.reset(Idx);
      }

      if (isa<CallBase>(&I)) {
        SeenCall = true;
        BBUsesAcrossCall |= TmpUses;
      }
    }
    HasCalls[BB->getNumber()] = SeenCall;
    Worklist.insert(BB);
  }


  while (!Worklist.empty()) {
    auto *BB = Worklist.back();
    Worklist.pop_back();

    auto &BBLiveOut = LiveOut[BB];
    auto &BBLiveOutAcrossCall = LiveOutAcrossCall[BB];

    BBLiveOut.reset();
    BBLiveOutAcrossCall.reset();

    for (BasicBlock *Succ : successors(BB)) {
      BBLiveOut |= LiveIn[Succ];
      BBLiveOutAcrossCall |= LiveInAcrossCall[Succ];
    }

    // Normal liveliness
    BitVector NewLiveIn = BBLiveOut;
    NewLiveIn |= Uses[BB];
    NewLiveIn.reset(Defs[BB]);


    // Liveliness specifically of call crossing
    BitVector NewLiveInAcrossCall = BBLiveOutAcrossCall;
    NewLiveInAcrossCall |= UsesAcrossCall[BB];
    NewLiveInAcrossCall.reset(Defs[BB]);

    if (HasCalls[BB->getNumber()]) {
      BitVector Tmp = BBLiveOut;
      Tmp.reset(DefsAfterAllCalls[BB]);
      NewLiveInAcrossCall |= Tmp;
    }


    if (NewLiveIn != LiveIn[BB] || NewLiveInAcrossCall != LiveInAcrossCall[BB]) {
      LiveIn[BB] = NewLiveIn;
      LiveInAcrossCall[BB] = NewLiveInAcrossCall;

      for (BasicBlock *Pred : predecessors(BB)) {
        Worklist.insert(Pred);
      }
    }
  }

  BitVector NeedsSpill(NumVRegs);
  for (auto &BB : F) {
    NeedsSpill |= UsesAcrossCall[&BB];
    NeedsSpill |= LiveOutAcrossCall[&BB];
  }

  // TODO: mark lifetimes and/or reuse alloca to reduce stack usage
  auto &entryBB = F.getEntryBlock();
  auto allocaPoint = entryBB.getFirstInsertionPt();

  for (Instruction *I : PotentialPointers) {
    if (!NeedsSpill[InstIdx[I]]) continue;

    Changed = true;

    auto *ai = new AllocaInst(I->getType(), 0, "stackSpill." + (I->hasName() ? I->getName() : "unnamedVreg"), allocaPoint);

    auto *store = new StoreInst(I, ai, true, ai->getAlign(), nullptr);

   std::optional<BasicBlock::iterator> After = I->getInsertionPointAfterDef();
   assert(After.has_value() && "Cannot spill value as it has no valid insertion point after it.");

    store->insertBefore(*After);
  }

  return Changed;
}
