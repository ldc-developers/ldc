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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/CFG.h"

using namespace llvm;

class LLVM_LIBRARY_VISIBILITY WasmPointersSpillLegacyPass
    : public FunctionPass {
  WasmPointersSpill pass;

public:
  static char ID; // Pass identification
  WasmPointersSpillLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override { return pass.run(F); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
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
  bool Changed = false;

  // Make sure we don't have any dead code that'll be missed by RPOT
  // and cause issues down the line.
  Changed = EliminateUnreachableBlocks(F);

#if LLVM_VERSION_MAJOR >= 20
  unsigned BBCount = F.getMaxBlockNumber();
#else
  DenseMap<BasicBlock *, unsigned> BBIdx;
  unsigned BBCount = 0;
  for (auto &BB : F) {
    BBIdx[&BB] = BBCount++;
  }
#endif

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
          if (Alloca)
            ValuesStoredInAlloca[Alloca].insert(Store->getValueOperand());
        }

        if (isa<AllocaInst>(I)) // the result of `alloca` can't be a GC pointer
          continue;

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

            if (WorklistSeen.insert(&I).second)
              Worklist.push_back(&I);
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
      if (TypeContainsPointers(V->getType()) && !isa<LoadInst>(V))
        return;

      if (auto *OpInst = dyn_cast<Instruction>(V)) {
        if (WorklistSeen.insert(OpInst).second)
          Worklist.push_back(OpInst);
      }
    };

    while (!Worklist.empty()) {
      auto *I = Worklist.back();
      Worklist.pop_back();

      if (I->getType()->isVoidTy()) // we don't care about non-values
        continue;
      if (isa<AllocaInst>(I)) // the result of `alloca` can't be a GC pointer
        continue;

      // small integers (and bools) can't hold values large enough to
      // be in the GC heap; assuming we have at least 64K of stack space + data
      if (I->getType()->isIntegerTy() &&
          I->getType()->getScalarSizeInBits() <= 16)
        continue;

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
      if (isa<CallBase>(I) && !isa<IntrinsicInst>(I))
        continue;

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

  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : make_range(RPOT.begin(), RPOT.end())) {
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
      // Don't mark PHI operands as uses (cause those will) propogate
      // to all predecessors, which is incorrect.
      if (!isa<PHINode>(&I))
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
        if (!SeenCall)
          BBDefsAfterAllCalls.set(Idx);
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

      for (Instruction &I : *Succ) {
        auto *PHI = dyn_cast<PHINode>(&I);
        if (!PHI)
          break;

        int IncomingIdx = PHI->getBasicBlockIndex(BB);
        if (IncomingIdx < 0)
          continue;

        Value *IncomingValue = PHI->getIncomingValue(IncomingIdx);
        if (auto *IncomingInst = dyn_cast<Instruction>(IncomingValue)) {
          if (InstIdx.contains(IncomingInst)) {
            unsigned Idx = InstIdx[IncomingInst];
            BBLiveOut.set(Idx);

            if (LiveInAcrossCall[Succ][Idx])
              BBLiveOutAcrossCall.set(Idx);
          }
        }
      }
    }

    // Normal liveliness
    BitVector NewLiveIn = BBLiveOut;
    NewLiveIn |= Uses[BB];
    NewLiveIn.reset(Defs[BB]);

    // Liveliness specifically of call crossing
    BitVector NewLiveInAcrossCall = BBLiveOutAcrossCall;

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

    NewLiveInAcrossCall |= UsesAcrossCall[BB];
    NewLiveInAcrossCall.reset(Defs[BB]);

    if (NewLiveIn != LiveIn[BB] ||
        NewLiveInAcrossCall != LiveInAcrossCall[BB]) {
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

#if LLVM_VERSION_MAJOR >= 20
    unsigned BBNum = BB.getNumber();
#else
    unsigned BBNum = BBIdx[&BB];
#endif
    if (HasCalls[BBNum]) {
      BitVector Tmp = LiveOut[&BB];
      Tmp.reset(DefsAfterAllCalls[&BB]);
      NeedsSpill |= Tmp;
    }
  }

  auto &entryBB = F.getEntryBlock();
  auto allocaPoint = entryBB.getFirstInsertionPt();

  auto &DL = F.getParent()->getDataLayout();

#if LLVM_VERSION_MAJOR >= 20
  Function *LifetimeStartFn = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::lifetime_start,
      {DL.getAllocaPtrType(F.getContext())});
  Function *LifetimeEndFn =
      Intrinsic::getOrInsertDeclaration(F.getParent(), Intrinsic::lifetime_end,
                                        {DL.getAllocaPtrType(F.getContext())});
#else
  Function *LifetimeStartFn =
      Intrinsic::getDeclaration(F.getParent(), Intrinsic::lifetime_start,
                                {DL.getAllocaPtrType(F.getContext())});
  Function *LifetimeEndFn =
      Intrinsic::getDeclaration(F.getParent(), Intrinsic::lifetime_end,
                                {DL.getAllocaPtrType(F.getContext())});
#endif

  IRBuilder<> Builder(&entryBB, allocaPoint);

  SmallVector<Instruction *> SpillPointers;
  SmallVector<AllocaInst *> SpillAllocas;
#if LLVM_VERSION_MAJOR < 22
  SmallVector<ConstantInt *> SpillSizes;
#endif

  Builder.SetInsertPoint(allocaPoint);

#if LLVM_VERSION_MAJOR < 22
  IntegerType *I64Type = IntegerType::get(F.getContext(), 64);
#endif
  for (Instruction *I : PotentialPointers) {
    unsigned Idx = InstIdx[I];
    if (!NeedsSpill[Idx])
      continue;

    Changed = true;

    SpillAllocas.push_back(Builder.CreateAlloca(
        I->getType(), nullptr,
        "stackSpill." + (I->hasName() ? I->getName() : "unnamedVreg")));
    SpillPointers.push_back(I);

#if LLVM_VERSION_MAJOR < 22
    SpillSizes.push_back(
        ConstantInt::get(I64Type, DL.getTypeAllocSize(I->getType())));
#endif
  }

  unsigned NextSpillPointerIdx = 0;

  SmallPtrSet<BasicBlock *, 8> NewBlocks;
  // do `lifetime.end` first
  for (Instruction *I : SpillPointers) {
    unsigned Idx = InstIdx[I];
#if LLVM_VERSION_MAJOR < 22
    ConstantInt *size = SpillSizes[NextSpillPointerIdx];
#endif
    AllocaInst *ai = SpillAllocas[NextSpillPointerIdx++];
    for (auto &BB : F) {
      if (NewBlocks.contains(&BB))
        continue;

      if (!LiveInAcrossCall[&BB][Idx] && !Defs[&BB][Idx])
        continue;

      if (LiveOutAcrossCall[&BB][Idx]) {
        // We want to find the specific edges that the spill dies across
        // (if any), split them (if needed), and insert a lifetime.end.
        // We add redundant branching in the IR, but tighten the lifetime.
        //
        // Hopefully optimizations during codegen will dissolve the empty BB
        // after the lifetime.end is lowered away.

        SmallPtrSet<BasicBlock *, 4> EdgesToKillOn;

        for (BasicBlock *Succ : successors(&BB)) {
          if (NewBlocks.contains(Succ))
            continue;

          // Rather than going down the rabbit hole of trying to find a spot
          // to place the lifetime.end, we simply omit it for EH pad targets
          // (e.g. the unwind target of `invoke`). Worst case is the stack
          // alloca lives longer than necessary. Better than having to e.g.
          // duplicate the whole EH funclet to slot it in.
          if (!LiveInAcrossCall[Succ][Idx] && !Succ->isEHPad())
            EdgesToKillOn.insert(Succ);
        }

        for (BasicBlock *Succ : EdgesToKillOn) {
          { // SplitEdge, but tweaked to merge identical edges, and only
            // split when needed (otherwise just set the insert point).
            unsigned SuccNum = GetSuccessorNumber(&BB, Succ);

            Instruction *LatchTerm = BB.getTerminator();

            CriticalEdgeSplittingOptions Options =
                CriticalEdgeSplittingOptions().setPreserveLCSSA();
            Options.setMergeIdenticalEdges();

            if (isCriticalEdge(LatchTerm, SuccNum,
                               Options.MergeIdenticalEdges)) {
              // If this is a critical edge, let SplitKnownCriticalEdge do it.
              BasicBlock *SplitBB =
                  SplitKnownCriticalEdge(LatchTerm, SuccNum, Options,
                                         ai->getName() + ".lifetimeEnd.bb");
              NewBlocks.insert(SplitBB);
              Builder.SetInsertPoint(SplitBB->begin());
            } else {
              // If the edge isn't critical, then BB has a single successor or
              // Succ has a single pred. Set the insert point appropriately.

              if (BasicBlock *SP = Succ->getUniquePredecessor()) {
                // If the successor only has a single pred, insert at the top
                // of the successor block.
                assert(SP == &BB && "CFG broken");
                (void)SP;

                Builder.SetInsertPoint(Succ->getFirstInsertionPt());
              } else {
                /// If it is LiveOutAcrossCall, it must be LiveInAcrossCall
                /// in one of the successors. If there is only one unique
                /// successor, it must be live-in across that edge, which means
                /// it won't be killed.
                assert(0 && "VReg is LiveOutAcrossCall, has only on succ, yet "
                            "dies across said only edge?");
              }
            }
          }

          Builder.CreateCall(LifetimeEndFn,
#if LLVM_VERSION_MAJOR >= 22
                             { ai }
#else
                             {size, ai}
#endif
          );
        }
      } else {
        if (LiveOut[&BB][Idx]) {
          // There are no more calls between the end of this block and
          // its future use (if any), but it IS still used, so keep the
          // spill alive until the final call in this block.
#if LLVM_VERSION_MAJOR >= 20
          unsigned BBNum = BB.getNumber();
#else
          unsigned BBNum = BBIdx[&BB];
#endif

          // If it is Def in this block, then for it to spill it must
          // have a UseAcrossCall in this block (and thus BB has calls), OR
          // it is LiveOutAcrossCall (handled above) because a future block
          // has a call before the reg is killed.
          //
          // If not Def, then it must be LiveInAcrossCall to be considered for
          // killing here. Which again means either we have a relavent call,
          // or a successor does (making it LiveOutAcrossCall).
          assert(HasCalls[BBNum] && "VReg is spilled, is LiveOut, BB has no "
                                    "calls, yet it dies here?");

          for (Instruction &IterI : make_range(BB.rbegin(), BB.rend())) {
            if (isa<CallBase>(&IterI)) {
              if (auto *Invoke = dyn_cast<InvokeInst>(&IterI)) {
                Builder.SetInsertPoint(
                    Invoke->getNormalDest()->getFirstInsertionPt());

                // This doesn't place a lifetime.end down the unwind path,
                // as there is no guaranteed spot to put it. Can't split the
                // unwind edge, and catchswitch (and the downstream catchpad)
                // may have multiple predecessors. Have to accept the lifetime
                // leak.
              } else {
                assert(!IterI.isTerminator());
                Builder.SetInsertPoint(std::next(IterI.getIterator()));
              }
              break;
            }
          }
        } else if (UsesAcrossCall[&BB][Idx]) {
          // No need to keep it around past this block.
          // End its lifetime after the last call before its last use.
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
                Builder.SetInsertPoint(
                    Invoke->getNormalDest()->getFirstInsertionPt());
              } else {
                assert(!IterI.isTerminator());
                Builder.SetInsertPoint(std::next(IterI.getIterator()));
              }
              break;
            }
          }
        } else {
          assert(0);
        }

        Builder.CreateCall(LifetimeEndFn,
#if LLVM_VERSION_MAJOR >= 22
                           { ai }
#else
                           {size, ai}
#endif
        );
      }
    }
  }

  // then insert `lifetime.start` and store,
  // skipping after any `lifetime.end`
  NextSpillPointerIdx = 0;
  for (Instruction *I : SpillPointers) {
#if LLVM_VERSION_MAJOR < 22
    ConstantInt *size = SpillSizes[NextSpillPointerIdx];
#endif
    AllocaInst *ai = SpillAllocas[NextSpillPointerIdx++];

    BasicBlock::iterator SpillInsertPoint;

    assert(!I->isTerminator() || isa<InvokeInst>(I));

    BasicBlock *BB = I->getParent();

    if (auto *Invoke = dyn_cast<InvokeInst>(I)) {
      // The result of `invoke` will never be spilled if its normal
      // destination/successor has multiple predecessors.
      //
      // The only way the value could be used in such sucessors would be in
      // a PHI. And if there is a PHI (which must be first-thing in a block),
      // and this `invoke` is a terminator, there's no way to have a call
      // that uses the `invoke` result before its use in the PHI.
      assert(Invoke->getNormalDest()->hasNPredecessors(1));
      SpillInsertPoint = Invoke->getNormalDest()->getFirstInsertionPt();
    } else if (isa<PHINode>(I)) {
      if (isa<CatchSwitchInst>(BB->getTerminator())) {
        // Having a catchswitch for a terminator means no instructions other
        // than PHI can be here. And since all successors must be catchpad
        // we can't split the edge either.
        //
        // However, Wasm EH associates each catchswitch with a single catchpad.
        // So spill at the start of that single successor.

        BasicBlock *Succ = BB->getSingleSuccessor();
        assert(Succ && "catchswitch with multiple catchpads?");

        SpillInsertPoint = Succ->getFirstInsertionPt();
      } else {
        // After all the PHI
        SpillInsertPoint = BB->getFirstInsertionPt();
      }
    } else {
      SpillInsertPoint = std::next(I->getIterator());
    }

    while (auto *Intrinsic = dyn_cast<IntrinsicInst>(&*SpillInsertPoint)) {
      if (Intrinsic->getIntrinsicID() != Intrinsic::lifetime_end)
        break;
      SpillInsertPoint = std::next(SpillInsertPoint);
    }

    Builder.SetInsertPoint(SpillInsertPoint);
    Builder.CreateCall(LifetimeStartFn,
#if LLVM_VERSION_MAJOR >= 22
                       { ai }
#else
                       {size, ai}
#endif
    );
    Builder.CreateStore(I, ai, true);
  }

  return Changed;
}
