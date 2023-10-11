//===-- GarbageCollect2Stack.cpp - Promote or remove GC allocations -------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This file attempts to turn allocations on the garbage-collected heap into
// stack allocations.
//
//===----------------------------------------------------------------------===//

#include "gen/attributes.h"
#include "metadata.h"
#include "gen/passes/GarbageCollect2Stack.h"
#include "gen/runtime.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "dgc2stack"

using namespace llvm;

STATISTIC(NumGcToStack, "Number of calls promoted to constant-size allocas");
STATISTIC(NumToDynSize,
          "Number of calls promoted to dynamically-sized allocas");
STATISTIC(NumDeleted,
          "Number of GC calls deleted because the return value was unused");

static cl::opt<unsigned>
    SizeLimit("dgc2stack-size-limit", cl::ZeroOrMore, cl::Hidden,
              cl::init(1024),
              cl::desc("Require allocs to be smaller than n bytes to be "
                       "promoted, 0 to ignore."));

struct G2StackAnalysis {
  const llvm::DataLayout &DL;
  const llvm::Module &M;
  llvm::CallGraph *CG;
  llvm::CallGraphNode *CGNode;

  llvm::Type *getTypeFor(llvm::Value *typeinfo, unsigned OperandNo) const;
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void EmitMemSet(IRBuilder<> &B, Value *Dst, Value *Val, Value *Len,
                const G2StackAnalysis &A) {
  Dst = B.CreateBitCast(Dst, PointerType::getUnqual(B.getInt8Ty()));

  MaybeAlign Align(1);

  auto CS = B.CreateMemSet(Dst, Val, Len, Align, false /*isVolatile*/);
  if (A.CGNode) {
    auto calledFunc = CS->getCalledFunction();
    A.CGNode->addCalledFunction(CS, A.CG->getOrInsertFunction(calledFunc));
  }
}

static void EmitMemZero(IRBuilder<> &B, Value *Dst, Value *Len,
                        const G2StackAnalysis &A) {
  EmitMemSet(B, Dst, ConstantInt::get(B.getInt8Ty(), 0), Len, A);
}

//===----------------------------------------------------------------------===//
// Helpers for specific types of GC calls.
//===----------------------------------------------------------------------===//

//namespace {

Value* FunctionInfo::promote(CallBase *CB, IRBuilder<> &B, const G2StackAnalysis &A) {
  NumGcToStack++;

  auto &BB = CB->getCaller()->getEntryBlock();
  Instruction *Begin = &(*BB.begin());

  // FIXME: set alignment on alloca?
  return new AllocaInst(Ty,
      BB.getModule()->getDataLayout().getAllocaAddrSpace(),
      ".nongc_mem", Begin);
}

static bool isKnownLessThan(Value *Val, uint64_t Limit, const G2StackAnalysis &A) {
  unsigned BitsLimit = Log2_64(Limit);

  // LLVM's alloca ueses an i32 for the number of elements.
  BitsLimit = std::min(BitsLimit, 32U);

  const IntegerType *SizeType = dyn_cast<IntegerType>(Val->getType());
  if (!SizeType) {
    return false;
  }
  unsigned Bits = SizeType->getBitWidth();

  if (Bits > BitsLimit) {
    APInt Mask = APInt::getLowBitsSet(Bits, BitsLimit);
    Mask.flipAllBits();
    KnownBits Known(Bits);
    computeKnownBits(Val, Known, A.DL);
    if ((Known.Zero & Mask) != Mask) {
      return false;
    }
  }

  return true;
}

bool TypeInfoFI::analyze(CallBase *CB, const G2StackAnalysis &A) {
  Value *TypeInfo = CB->getArgOperand(TypeInfoArgNr);
  Ty = A.getTypeFor(TypeInfo, 0);
  if (!Ty) {
    return false;
  }
  return A.DL.getTypeAllocSize(Ty) < SizeLimit;
}

bool ArrayFI::analyze(CallBase *CB, const G2StackAnalysis &A) {
  if (!TypeInfoFI::analyze(CB, A)) {
    return false;
  }

  arrSize = CB->getArgOperand(ArrSizeArgNr);
  Value *TypeInfo = CB->getArgOperand(TypeInfoArgNr);
  Ty = A.getTypeFor(TypeInfo, 1);
  // If the user explicitly disabled the limits, don't even check
  // whether the element count fits in 32 bits. This could cause
  // miscompilations for humongous arrays, but as the value "range"
  // (set bits) inference algorithm is rather limited, this is
  // useful for experimenting.
  if (SizeLimit > 0) {
    uint64_t ElemSize = A.DL.getTypeAllocSize(Ty);
    if (!isKnownLessThan(arrSize, SizeLimit / ElemSize, A)) {
      return false;
    }
  }

  return true;
}

Value* ArrayFI::promote(CallBase *CB, IRBuilder<> &B, const G2StackAnalysis &A) {
  IRBuilder<> Builder(B.GetInsertBlock(), B.GetInsertPoint());

  // If the allocation is of constant size it's best to put it in the
  // entry block, so do so if we're not already there.
  // For dynamically-sized allocations it's best to avoid the overhead
  // of allocating them if possible, so leave those where they are.
  // While we're at it, update statistics too.
  if (isa<Constant>(arrSize)) {
    BasicBlock &Entry = CB->getCaller()->getEntryBlock();
    if (Builder.GetInsertBlock() != &Entry) {
      Builder.SetInsertPoint(&Entry, Entry.begin());
    }
    NumGcToStack++;
  } else {
    NumToDynSize++;
  }

  // Convert array size to 32 bits if necessary
  Value *count = Builder.CreateIntCast(arrSize, Builder.getInt32Ty(), false);
  AllocaInst *alloca =
      Builder.CreateAlloca(Ty, count, ".nongc_mem"); // FIXME: align?

  if (Initialized) {
    // For now, only zero-init is supported.
    uint64_t size = A.DL.getTypeStoreSize(Ty);
    Value *TypeSize = ConstantInt::get(arrSize->getType(), size);
    // The initialization must be put at the original source variable
    // definition location, because it could be in a loop and because
    // of lifetime start-end annotation.
    Value *Size = B.CreateMul(TypeSize, arrSize);
    EmitMemZero(B, alloca, Size, A);
  }

  if (ReturnType == ReturnType::Array) {
    Value *arrStruct = llvm::UndefValue::get(CB->getType());
    arrStruct = Builder.CreateInsertValue(arrStruct, arrSize, 0);
    Value *memPtr =
        Builder.CreateBitCast(alloca, PointerType::getUnqual(B.getInt8Ty()));
    arrStruct = Builder.CreateInsertValue(arrStruct, memPtr, 1);
    return arrStruct;
  }

  return alloca;
}
bool AllocClassFI::analyze(CallBase *CB, const G2StackAnalysis &A) {
  if (CB->arg_size() != 1) {
    return false;
  }
  Value *arg = CB->getArgOperand(0)->stripPointerCasts();
  GlobalVariable *ClassInfo = dyn_cast<GlobalVariable>(arg);
  if (!ClassInfo) {
    return false;
  }

  const auto metaname = getMetadataName(CD_PREFIX, ClassInfo);

  NamedMDNode *meta = A.M.getNamedMetadata(metaname);
  if (!meta) {
    return false;
  }

  MDNode *node = static_cast<MDNode *>(meta->getOperand(0));
  if (!node || node->getNumOperands() != CD_NumFields) {
    return false;
  }

  // Inserting destructor calls is not implemented yet, so classes
  // with destructors are ignored for now.
  auto hasDestructor =
      mdconst::dyn_extract<Constant>(node->getOperand(CD_Finalize));
  if (hasDestructor == nullptr ||
      hasDestructor != ConstantInt::getFalse(A.M.getContext())) {
    return false;
  }

  Ty = mdconst::dyn_extract<Constant>(node->getOperand(CD_BodyType))
           ->getType();
  return A.DL.getTypeAllocSize(Ty) < SizeLimit;
}
bool UntypedMemoryFI::analyze(CallBase *CB, const G2StackAnalysis &A) {
  if (CB->arg_size() < SizeArgNr + 1) {
    return false;
  }

  SizeArg = CB->getArgOperand(SizeArgNr);

  // If the user explicitly disabled the limits, don't even check
  // whether the allocated size fits in 32 bits. This could cause
  // miscompilations for humongous allocations, but as the value
  // "range" (set bits) inference algorithm is rather limited, this
  // is useful for experimenting.
  if (SizeLimit > 0) {
    if (!isKnownLessThan(SizeArg, SizeLimit, A)) {
      return false;
    }
  }

  // Should be i8.
  Ty = llvm::Type::getInt8Ty(CB->getContext());
  return true;
}
Value* UntypedMemoryFI::promote(CallBase *CB, IRBuilder<> &B, const G2StackAnalysis &A) {
  // If the allocation is of constant size it's best to put it in the
  // entry block, so do so if we're not already there.
  // For dynamically-sized allocations it's best to avoid the overhead
  // of allocating them if possible, so leave those where they are.
  // While we're at it, update statistics too.
  const IRBuilderBase::InsertPointGuard savedInsertPoint(B);
  if (isa<Constant>(SizeArg)) {
    BasicBlock &Entry = CB->getCaller()->getEntryBlock();
    if (B.GetInsertBlock() != &Entry) {
      B.SetInsertPoint(&Entry, Entry.begin());
    }
    NumGcToStack++;
  } else {
    NumToDynSize++;
  }

  // Convert array size to 32 bits if necessary
  Value *count = B.CreateIntCast(SizeArg, B.getInt32Ty(), false);
  AllocaInst *alloca =
      B.CreateAlloca(Ty, count, ".nongc_mem"); // FIXME: align?

  return B.CreateBitCast(alloca, CB->getType());
}
//}

//===----------------------------------------------------------------------===//
// GarbageCollect2Stack Pass Implementation
//===----------------------------------------------------------------------===//

//namespace {

class LLVM_LIBRARY_VISIBILITY GarbageCollect2StackLegacyPass : public FunctionPass {

  bool doInitialization(Module &M) override {
    this->pass.M = &M;
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<CallGraphWrapperPass>();
  }
  bool runOnFunction(Function &F) override {

    auto getDT = [&]() -> DominatorTree& {
      return getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    };

    auto getCG = [&]() -> CallGraph* {
      CallGraphWrapperPass* const CGPass = getAnalysisIfAvailable<CallGraphWrapperPass>();
      return CGPass ? &CGPass->getCallGraph() : nullptr;
    };

    return pass.run(F, getDT, getCG);
  }
  StringRef getPassName() const override { return GarbageCollect2Stack::getPassName(); }

public:
  GarbageCollect2StackLegacyPass() :
      FunctionPass(ID), pass() {}
  static char ID; // Pass identification
  GarbageCollect2Stack pass;
};

char GarbageCollect2StackLegacyPass::ID = 0;
//} // end anonymous namespace.
static RegisterPass<GarbageCollect2StackLegacyPass>
    X("dgc2stack", "Promote (GC'ed) heap allocations to stack");

// Public interface to the pass.
FunctionPass *createGarbageCollect2Stack() {
  return new GarbageCollect2StackLegacyPass();
}

GarbageCollect2Stack::GarbageCollect2Stack()
    : AllocMemoryT(ReturnType::Pointer, 0),
      NewArrayU(ReturnType::Array, 0, 1, false),
      NewArrayT(ReturnType::Array, 0, 1, true), AllocMemory(0) {
}

static void RemoveCall(CallBase *CB, const G2StackAnalysis &A) {
  // For an invoke instruction, we insert a branch to the normal target BB
  // immediately before it. Ideally, we would find a way to not invalidate
  // the dominator tree here.
  if (auto Invoke = dyn_cast<InvokeInst>(static_cast<Instruction *>(CB))) {
    BranchInst::Create(Invoke->getNormalDest(), Invoke);
    Invoke->getUnwindDest()->removePredecessor(CB->getParent());
  }

  // Remove the runtime call.
  if (A.CGNode) {
    A.CGNode->removeCallEdgeFor(*CB);
  }
  static_cast<Instruction *>(CB)->eraseFromParent();
}

static bool
isSafeToStackAllocateArray(BasicBlock::iterator Alloc, DominatorTree &DT,
                           SmallVector<CallInst *, 4> &RemoveTailCallInsts);
static bool
isSafeToStackAllocate(BasicBlock::iterator Alloc, Value *V, DominatorTree &DT,
                      SmallVector<CallInst *, 4> &RemoveTailCallInsts);

/// runOnFunction - Top level algorithm.
///
bool GarbageCollect2Stack::run(Function &F, std::function<DominatorTree& ()> getDT, std::function<CallGraph* ()> getCG) {
  LLVM_DEBUG(errs() << "\nRunning -dgc2stack on function " << F.getName() << '\n');
  DominatorTree& DT = getDT();
  CallGraph* CG = getCG();
  const DataLayout &DL = F.getParent()->getDataLayout();
  CallGraphNode *CGNode = CG ? (*CG)[&F] : nullptr;
  G2StackAnalysis A = {DL, *F.getParent(), CG, CGNode};

  BasicBlock &Entry = F.getEntryBlock();

  IRBuilder<> AllocaBuilder(&Entry, Entry.begin());

  bool Changed = false;
  for (auto &BB : F) {
    for (auto I = BB.begin(), E = BB.end(); I != E;) {
      auto originalI = I;

      // Ignore non-calls.
      Instruction *Inst = &(*(I++));
      auto CB = dyn_cast<CallBase>(Inst);
      if (!CB) {
        continue;
      }

      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CB->getCalledFunction();
      if (Callee == nullptr || !Callee->isDeclaration() ||
          !Callee->hasExternalLinkage()) {
        continue;
      }


      //FunctionInfo *info = OMI->getValue();
      FunctionInfo *info = StringSwitch<FunctionInfo*>(Callee->getName())
     .Case("_d_allocmemoryT", &AllocMemoryT)
     .Case("_d_newarrayU",    &NewArrayU)
     .Case("_d_newarrayT",    &NewArrayT)
     .Case("_d_allocclass",   &AllocClass)
     .Case("_d_allocmemory",  &AllocMemory)
     .Default(nullptr);

      // Ignore unknown calls.
      if (!info) {
        continue;
      }

      if (static_cast<Instruction *>(CB)->use_empty()) {
        Changed = true;
        NumDeleted++;
        RemoveCall(CB, A);
        continue;
      }

      LLVM_DEBUG(errs() << "GarbageCollect2Stack inspecting: " << *CB);

      if ( !info->analyze(CB, A)) {
        continue;
      }

      SmallVector<CallInst *, 4> RemoveTailCallInsts;
      if (info->ReturnType == ReturnType::Array) {
        if (!isSafeToStackAllocateArray(originalI, DT, RemoveTailCallInsts)) {
          continue;
        }
      } else {
        if (!isSafeToStackAllocate(originalI, CB, DT, RemoveTailCallInsts)) {
          continue;
        }
      }

      // Let's alloca this!
      Changed = true;

      // First demote tail calls which use the value so there IR is never
      // in an invalid state.
      for (auto i : RemoveTailCallInsts) {
        i->setTailCall(false);
      }

      IRBuilder<> Builder(&BB, originalI);
      Value *newVal = info->promote(CB, Builder, A);

      LLVM_DEBUG(errs() << "Promoted to: " << *newVal);

      // Make sure the type is the same as it was before, and replace all
      // uses of the runtime call with the alloca.
      if (newVal->getType() != CB->getType()) {
        newVal = Builder.CreateBitCast(newVal, CB->getType());
      }
      static_cast<Instruction *>(CB)->replaceAllUsesWith(newVal);

      RemoveCall(CB, A);
    }
  }

  return Changed;
}

llvm::Type *G2StackAnalysis::getTypeFor(Value *typeinfo, unsigned OperandNo) const {
  GlobalVariable *ti_global =
      dyn_cast<GlobalVariable>(typeinfo->stripPointerCasts());
  if (!ti_global) {
    return nullptr;
  }

  const auto metaname = getMetadataName(TD_PREFIX, ti_global);

  NamedMDNode *meta = M.getNamedMetadata(metaname);
  if (!meta || (meta->getNumOperands() != 1 && meta->getNumOperands() != 2) ) {
    return nullptr;
  }

  MDNode *node = meta->getOperand(OperandNo);
  return llvm::cast<llvm::ConstantAsMetadata>(node->getOperand(0))->getType();
}

/// Returns whether Def is used by any instruction that is reachable from Alloc
/// (without executing Def again).
static bool mayBeUsedAfterRealloc(Instruction *Def, BasicBlock::iterator Alloc,
                                  DominatorTree &DT) {
  LLVM_DEBUG(errs() << "### mayBeUsedAfterRealloc()\n" << *Def << *Alloc);

  // If the definition isn't used it obviously won't be used after the
  // allocation.
  // If it does not dominate the allocation, there's no way for it to be used
  // without going through Def again first, since the definition couldn't
  // dominate the user either.
  if (Def->use_empty() || !DT.dominates(Def, &(*Alloc))) {
    LLVM_DEBUG(errs() << "### No uses or does not dominate allocation\n");
    return false;
  }

  LLVM_DEBUG(errs() << "### Def dominates Alloc\n");

  BasicBlock *DefBlock = Def->getParent();
  BasicBlock *AllocBlock = Alloc->getParent();

  // Create a set of users and one of blocks containing users.
  SmallSet<User *, 16> Users;
  SmallSet<BasicBlock *, 16> UserBlocks;
  for (Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();
       UI != UE; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    LLVM_DEBUG(errs() << "USER: " << *User);
    BasicBlock *UserBlock = User->getParent();

    // This dominance check is not performed if they're in the same block
    // because it will just walk the instruction list to figure it out.
    // We will instead do that ourselves in the first iteration (for all
    // users at once).
    if (AllocBlock != UserBlock && DT.dominates(AllocBlock, UserBlock)) {
      // There's definitely a path from alloc to this user that does not
      // go through Def, namely any path that ends up in that user.
      LLVM_DEBUG(errs() << "### Alloc dominates user " << *User);
      return true;
    }

    // Phi nodes are checked separately, so no need to enter them here.
    if (!isa<PHINode>(User)) {
      Users.insert(User);
      UserBlocks.insert(UserBlock);
    }
  }

  // Contains first instruction of block to inspect.
  typedef std::pair<BasicBlock *, BasicBlock::iterator> StartPoint;
  SmallVector<StartPoint, 16> Worklist;
  // Keeps track of successors that have been added to the work list.
  SmallSet<BasicBlock *, 16> Visited;

  // Start just after the allocation.
  // Note that we don't insert AllocBlock into the Visited set here so the
  // start of the block will get inspected if it's reachable.
  BasicBlock::iterator Start = Alloc;
  ++Start;
  Worklist.push_back(StartPoint(AllocBlock, Start));

  while (!Worklist.empty()) {
    StartPoint sp = Worklist.pop_back_val();
    BasicBlock *B = sp.first;
    BasicBlock::iterator BBI = sp.second;
    // BBI is either just after the allocation (in the first iteration)
    // or just after the last phi node in B (in subsequent iterations) here.

    // This whole 'if' is just a way to avoid performing the inner 'for'
    // loop when it can be determined not to be necessary, avoiding
    // potentially expensive walks of the instruction list.
    // It should be equivalent to just doing the loop.
    if (UserBlocks.count(B)) {
      if (B != DefBlock && B != AllocBlock) {
        // This block does not contain the definition or the allocation,
        // so any user in this block is definitely reachable without
        // finding either the definition or the allocation.
        LLVM_DEBUG(errs() << "### Block " << B->getName()
                     << " contains a reachable user\n");
        return true;
      }
      // We need to walk the instructions in the block to see whether we
      // reach a user before we reach the definition or the allocation.
      for (BasicBlock::iterator E = B->end(); BBI != E; ++BBI) {
        if (&*BBI == &*Alloc || &*BBI == Def) {
          break;
        }
        if (Users.count(&(*BBI))) {
          LLVM_DEBUG(errs() << "### Problematic user: " << *BBI);
          return true;
        }
      }
    } else if (B == DefBlock || (B == AllocBlock && BBI != Start)) {
      // There are no users in the block so the def or the allocation
      // will be encountered before any users though this path.
      // Skip to the next item on the worklist.
      continue;
    } else {
      // No users and no definition or allocation after the start point,
      // so just keep going.
    }

    // All instructions after the starting point in this block have been
    // accounted for. Look for successors to add to the work list.
    auto *Term = B->getTerminator();
    unsigned SuccCount = Term->getNumSuccessors();
    for (unsigned i = 0; i < SuccCount; i++) {
      BasicBlock *Succ = Term->getSuccessor(i);
      BBI = Succ->begin();
      // Check phi nodes here because we only care about the operand
      // coming in from this block.
      bool SeenDef = false;
      while (isa<PHINode>(BBI)) {
        if (Def == cast<PHINode>(BBI)->getIncomingValueForBlock(B)) {
          LLVM_DEBUG(errs() << "### Problematic phi user: " << *BBI);
          return true;
        }
        SeenDef |= (Def == &*BBI);
        ++BBI;
      }
      // If none of the phis we just looked at were the definition, we
      // haven't seen this block yet, and it's dominated by the def
      // (meaning paths through it could lead to users), add the block and
      // the first non-phi to the worklist.
      if (!SeenDef
          && Visited.insert(Succ).second
          && DT.dominates(DefBlock, Succ)) {
        Worklist.push_back(StartPoint(Succ, BBI));
      }
    }
  }
  // No users found in any block reachable from Alloc
  // without going through the definition again.
  return false;
}

/// Returns true if the GC call passed in is safe to turn into a stack
/// allocation.
///
/// This handles GC calls returning a D array instead of a raw pointer,
/// see isSafeToStackAllocate() for details.
bool isSafeToStackAllocateArray(
    BasicBlock::iterator Alloc, DominatorTree &DT,
    SmallVector<CallInst *, 4> &RemoveTailCallInsts) {
  assert(Alloc->getType()->isStructTy() && "Allocated array is not a struct?");
  Value *V = &(*Alloc);

  for (auto U : V->users()) {
    Instruction *User = dyn_cast<Instruction>(U);
    if (User == nullptr) {
      continue;
    }

    switch (User->getOpcode()) {
    case Instruction::ExtractValue: {
      ExtractValueInst *EVI = cast<ExtractValueInst>(User);

      assert(EVI->getAggregateOperand() == V);
      assert(EVI->getNumIndices() == 1);

      unsigned idx = EVI->getIndices()[0];
      if (idx == 0) {
        // This extract the length argument, irrelevant for our analysis.
        assert(EVI->getType()->isIntegerTy() &&
               "First array field not length?");
      } else {
        assert(idx == 1 && "Invalid array struct access.");
        if (!isSafeToStackAllocate(Alloc, EVI, DT, RemoveTailCallInsts)) {
          return false;
        }
      }
      break;
    }
    default:
      // We are super conservative here, the only thing we want to be able to
      // handle at this point is extracting len/ptr. More extensive analysis
      // could be added later.
      return false;
    }
  }

  // All uses examined - memory not captured.
  return true;
}

/// Returns true if the GC call passed in is safe to turn
/// into a stack allocation. This requires that the return value does not
/// escape from the function and no derived pointers are live at the call site
/// (i.e. if it's in a loop then the function can't use any pointer returned
/// from an earlier call after a new call has been made).
///
/// This is currently conservative where loops are involved: it can handle
/// simple loops, but returns false if any derived pointer is used in a
/// subsequent iteration.
///
/// Based on LLVM's PointerMayBeCaptured(), which only does escape analysis but
/// doesn't care about loops.
///
/// Alloc is the actual call to the runtime function, and V is the pointer to
/// the memory it returns (which might not be equal to Alloc in case of
/// functions returning D arrays).
///
/// If the value is used in a call instruction with the tail attribute set,
/// the attribute has to be removed before promoting the memory to the
/// stack. The affected instructions are added to RemoveTailCallInsts. If
/// the function returns false, these entries are meaningless.
bool isSafeToStackAllocate(BasicBlock::iterator Alloc, Value *V,
                           DominatorTree &DT,
                           SmallVector<CallInst *, 4> &RemoveTailCallInsts) {
  assert(isa<PointerType>(V->getType()) && "Allocated value is not a pointer?");

  SmallVector<Use *, 16> Worklist;
  SmallSet<Use *, 16> Visited;

  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end(); UI != UE;
       ++UI) {
    Use *U = &(*UI);
    Visited.insert(U);
    Worklist.push_back(U);
  }

  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    Instruction *I = cast<Instruction>(U->getUser());
    V = U->get();

    switch (I->getOpcode()) {
    case Instruction::Call:
    case Instruction::Invoke: {
      auto CB = llvm::cast<CallBase>(I);
      // Not captured if the callee is readonly, doesn't return a copy through
      // its return value and doesn't unwind (a readonly function can leak bits
      // by throwing an exception or not depending on the input value).
      if (CB->onlyReadsMemory() && CB->doesNotThrow() &&
          I->getType() == llvm::Type::getVoidTy(I->getContext())) {
        break;
      }

      // Not captured if only passed via 'nocapture' arguments.  Note that
      // calling a function pointer does not in itself cause the pointer to
      // be captured.  This is a subtle point considering that (for example)
      // the callee might return its own address.  It is analogous to saying
      // that loading a value from a pointer does not cause the pointer to be
      // captured, even though the loaded value might be the pointer itself
      // (think of self-referential objects).
      auto B = CB->arg_begin(), E = CB->arg_end();
      for (auto A = B; A != E; ++A) {
        if (A->get() == V) {
          if (!CB->paramHasAttr(A - B, llvm::Attribute::AttrKind::NoCapture)) {
            // The parameter is not marked 'nocapture' - captured.
            return false;
          }

          if (auto call = dyn_cast<CallInst>(static_cast<Instruction *>(CB))) {
            if (call->isTailCall()) {
              RemoveTailCallInsts.push_back(call);
            }
          }
        }
      }
      // Only passed via 'nocapture' arguments, or is the called function - not
      // captured.
      break;
    }
    case Instruction::Load:
      // Loading from a pointer does not cause it to be captured.
      break;
    case Instruction::Store:
      if (V == I->getOperand(0)) {
        // Stored the pointer - it may be captured.
        return false;
      }
      // Storing to the pointee does not cause the pointer to be captured.
      break;
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Select:
      // It's not safe to stack-allocate if this derived pointer is live across
      // the original allocation.
      if (mayBeUsedAfterRealloc(I, Alloc, DT)) {
        return false;
      }

      // The original value is not captured via this if the new value isn't.
      for (Instruction::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {
        Use *U = &(*UI);
        if (Visited.insert(U).second) {
          Worklist.push_back(U);
        }
      }
      break;
    default:
      // Something else - be conservative and say it is captured.
      return false;
    }
  }

  // All uses examined - not captured or live across original allocation.
  return true;
}
