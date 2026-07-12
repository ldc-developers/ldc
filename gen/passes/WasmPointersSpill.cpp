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
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/ValueTracking.h"


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
#if LLVM_VERSION_MAJOR >= 20
  unsigned BBCount = F.getMaxBlockNumber();
#else
  DenseMap<BasicBlock *, unsigned> BBIdx;
  unsigned BBCount = 0;
  for (auto &BB : F) {
    BBIdx[&BB] = BBCount++;
  }
#endif

  bool Changed = false;

  // for SetVector determinstic iteration
  SmallSetVector<Instruction *, 0> PotentialPointers;

  {
    SmallPtrSet<Instruction *, 0> WorklistSeen;
    SmallVector<Instruction *> Worklist;

    DenseMap<AllocaInst *, SmallPtrSet<Value *, 0>> ValuesStoredInAlloca;

    for (auto &BB : F) {
      for (Instruction &I : BB) {
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          Value *StorePtr = Store->getPointerOperand();
          AllocaInst *Alloca = findAllocaForValue(StorePtr);
          if (Alloca) ValuesStoredInAlloca[Alloca].insert(Store->getValueOperand());
        }

        if (isa<AllocaInst>(I)) continue; // the result of `alloca` can't be a GC pointer

        if (TypeContainsPointers(I.getType())) {
          PotentialPointers.insert(&I);
          if (IntToPtrInst *IntToPtr = dyn_cast<IntToPtrInst>(&I)) {
            // if we have `inttoptr` it's transitive uses are themselves
            // potential pointers

            if (auto *OpInst = dyn_cast<Instruction>(IntToPtr->getOperand(0))) {
              WorklistSeen.insert(OpInst);
              Worklist.push_back(OpInst);
            }
          } else if (isa<LoadInst>(&I)) {
            // if we have a direct `load ptr`, treat it as
            // e.g. `load i32` and `inttoptr`
            //
            // if it's a load from an `alloca`, any stores
            // into said `alloca` might be hidden pointers

            if (WorklistSeen.insert(&I).second) Worklist.push_back(&I);
          }
        }
      }
    }


    auto markValue = [&](Value *V) {
      // If the op is/has a pointer, it's not a concern for
      // hidding values anymore; stop the back-tracking.
      //
      // However, we want to still trace loads back through memory
      // regardless to help catch `store`s with mismatching type.
      if (TypeContainsPointers(V->getType()) && !isa<LoadInst>(V)) return;

      if (auto *OpInst = dyn_cast<Instruction>(V)) {
        if (WorklistSeen.insert(OpInst).second) Worklist.push_back(OpInst);
      }
    };

    while (!Worklist.empty()) {
      auto *I = Worklist.back();
      Worklist.pop_back();

      if (I->getType()->isVoidTy()) continue; // we don't care about non-values
      if (isa<AllocaInst>(I)) continue; // the result of `alloca` can't be a GC pointer

      // small integers (and bools) can't hold values large enough to
      // be in the GC heap; assuming we have at least 64K of stack space + data
      if (I->getType()->isIntegerTy()
          && I->getType()->getScalarSizeInBits() <= 16) continue;

      PotentialPointers.insert(I);

      // trace `alloca` loads back through stores to the same
      if (auto *Load = dyn_cast<LoadInst>(I)) {
        Value *V = Load->getPointerOperand();
        AllocaInst *Alloca = findAllocaForValue(V);
        if (Alloca)
          for (Value *StoreValue : ValuesStoredInAlloca[Alloca])
            markValue(StoreValue);

        continue;
      }

      // We can safely assume the return of the function is
      // unrelated to any non-pointer operands.
      //
      // However, we can only assume that for D function calls
      // Not intrinsics LLVM might insert.
      if (isa<CallBase>(I) && !isa<IntrinsicInst>(I)) continue;

      for (Use &Op : I->operands()) {
        Value *V = Op.get();
        markValue(V);
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
  BitVector HasCalls(BBCount);

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

#if LLVM_VERSION_MAJOR >= 20
    unsigned BBNum = BB->getNumber();
#else
    unsigned BBNum = BBIdx[BB];
#endif
    HasCalls[BBNum] = SeenCall;
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

#if LLVM_VERSION_MAJOR >= 20
    unsigned BBNum = BB->getNumber();
#else
    unsigned BBNum = BBIdx[BB];
#endif
    if (HasCalls[BBNum]) {
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
  DenseMap<BasicBlock *, BitVector> SpillKills;
  for (auto &BB : F) {
    NeedsSpill |= UsesAcrossCall[&BB];
    NeedsSpill |= LiveOutAcrossCall[&BB];

    BitVector Tmp(NumVRegs);
    Tmp |= LiveInAcrossCall[&BB];
    Tmp |= Defs[&BB];
    Tmp.reset(LiveOutAcrossCall[&BB]);

    SpillKills[&BB] = Tmp;
  }

  auto &entryBB = F.getEntryBlock();
  auto allocaPoint = entryBB.getFirstInsertionPt();

#if LLVM_VERSION_MAJOR >= 20
  Function *LifetimeStartFn =
    Intrinsic::getOrInsertDeclaration(
      F.getParent(),
      Intrinsic::lifetime_start,
      {F.getParent()->getDataLayout().getAllocaPtrType(F.getContext())}
    );
  Function *LifetimeEndFn =
    Intrinsic::getOrInsertDeclaration(
      F.getParent(),
      Intrinsic::lifetime_end,
      {F.getParent()->getDataLayout().getAllocaPtrType(F.getContext())}
  );
#elif LLVM_VERSION_MAJOR >= 19
  Function *LifetimeStartFn =
    Intrinsic::getDeclaration(
      F.getParent(),
      Intrinsic::lifetime_start,
      {F.getParent()->getDataLayout().getAllocaPtrType(F.getContext())}
    );
  Function *LifetimeEndFn =
    Intrinsic::getDeclaration(
      F.getParent(),
      Intrinsic::lifetime_end,
      {F.getParent()->getDataLayout().getAllocaPtrType(F.getContext())}
  );
#else
  Function *LifetimeStartFn =
    Intrinsic::getDeclaration(
      F.getParent(),
      Intrinsic::lifetime_start
    );
  Function *LifetimeEndFn =
    Intrinsic::getDeclaration(
      F.getParent(),
      Intrinsic::lifetime_end
  );
#endif


  IRBuilder<> Builder(&entryBB, allocaPoint);

  SmallVector<Instruction *> SpillPointers;
  SmallVector<AllocaInst *> SpillAllocas;

  Builder.SetInsertPoint(allocaPoint);

  for (Instruction *I : PotentialPointers) {
    unsigned Idx = InstIdx[I];
    if (!NeedsSpill[Idx]) continue;

    Changed = true;

    SpillAllocas.push_back(Builder.CreateAlloca(
      I->getType(), nullptr,
      "stackSpill." + (I->hasName() ? I->getName() : "unnamedVreg")
    ));
    SpillPointers.push_back(I);
  }

  unsigned NextSpillPointerIdx = 0;

  // do `lifetime.end` first
  for (Instruction *I : SpillPointers) {
    unsigned Idx = InstIdx[I];
    AllocaInst *ai = SpillAllocas[NextSpillPointerIdx++];

    for (auto &BB : F) {
      if(!SpillKills[&BB][Idx]) continue;

      if (UsesAcrossCall[&BB][Idx]) {
        bool Found = false;
        for (Instruction &IterI : make_range(BB.rbegin(), BB.rend())) {
          for (Use &Op : IterI.operands()) {
            Value *V = Op.get();

            if (V == I) {
              Found = true;
              break;
            }
          }

          if (Found && isa<CallBase>(&IterI)) {
            if (auto *Invoke = dyn_cast<InvokeInst>(&IterI)) {
              Builder.SetInsertPoint(Invoke->getNormalDest()->getFirstInsertionPt());
            } else {
              assert(!IterI.isTerminator());
              Builder.SetInsertPoint(std::next(IterI.getIterator()));
            }
            break;
          }
        }
      } else {
        Builder.SetInsertPoint(&BB.front());
      }
      Builder.CreateCall(LifetimeEndFn, {ai});
    }
  }

  // then insert `lifetime.start` and store,
  // skipping after any `lifetime.end`
  NextSpillPointerIdx = 0;
  for (Instruction *I : SpillPointers) {
    AllocaInst *ai = SpillAllocas[NextSpillPointerIdx++];

    BasicBlock::iterator After;
    {
      std::optional<BasicBlock::iterator> MaybeAfter = I->getInsertionPointAfterDef();
      assert(MaybeAfter.has_value() && "Cannot spill value as it has no valid insertion point after it.");
      After = *MaybeAfter;
    }

    while (auto *Intrinsic = dyn_cast<IntrinsicInst>(&*After)) {
      if (Intrinsic->getIntrinsicID() != Intrinsic::lifetime_end) break;
      After = std::next(After);
    }


    Builder.SetInsertPoint(After);
    Builder.CreateCall(LifetimeStartFn, {ai});
    Builder.CreateStore(I, ai, true);
  }

  return Changed;
}
