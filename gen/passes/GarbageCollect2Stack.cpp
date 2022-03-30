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
#include "gen/passes/Passes.h"
#include "gen/runtime.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
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
#if LDC_LLVM_VER < 700
#define LLVM_DEBUG DEBUG
#endif

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

namespace {
struct Analysis {
  const DataLayout &DL;
  const Module &M;
  CallGraph *CG;
  CallGraphNode *CGNode;

  llvm::Type *getTypeFor(Value *typeinfo) const;
};
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void EmitMemSet(IRBuilder<> &B, Value *Dst, Value *Val, Value *Len,
                const Analysis &A) {
  Dst = B.CreateBitCast(Dst, PointerType::getUnqual(B.getInt8Ty()));

#if LDC_LLVM_VER >= 1000
  MaybeAlign Align(1);
#else
  unsigned Align = 1; 
#endif

  auto CS = B.CreateMemSet(Dst, Val, Len, Align, false /*isVolatile*/);
  if (A.CGNode) {
    auto calledFunc = CS->getCalledFunction();
    A.CGNode->addCalledFunction(CS, A.CG->getOrInsertFunction(calledFunc));
  }
}

static void EmitMemZero(IRBuilder<> &B, Value *Dst, Value *Len,
                        const Analysis &A) {
  EmitMemSet(B, Dst, ConstantInt::get(B.getInt8Ty(), 0), Len, A);
}

//===----------------------------------------------------------------------===//
// Helpers for specific types of GC calls.
//===----------------------------------------------------------------------===//

namespace {
namespace ReturnType {
enum Type {
  Pointer, /// Function returns a pointer to the allocated memory.
  Array    /// Function returns the allocated memory as an array slice.
};
}

class FunctionInfo {
protected:
  llvm::Type *Ty;

public:
  ReturnType::Type ReturnType;

  // Analyze the current call, filling in some fields. Returns true if
  // this is an allocation we can stack-allocate.
  virtual bool analyze(LLCallBasePtr CB, const Analysis &A) = 0;

  // Returns the alloca to replace this call.
  // It will always be inserted before the call.
  virtual Value *promote(LLCallBasePtr CB, IRBuilder<> &B, const Analysis &A) {
    NumGcToStack++;

    auto &BB = CB->getCaller()->getEntryBlock();
    Instruction *Begin = &(*BB.begin());

    // FIXME: set alignment on alloca?
    return new AllocaInst(Ty,
                          BB.getModule()->getDataLayout().getAllocaAddrSpace(),
                          ".nongc_mem", Begin);
  }

  explicit FunctionInfo(ReturnType::Type returnType) : ReturnType(returnType) {}
  virtual ~FunctionInfo() = default;
};

static bool isKnownLessThan(Value *Val, uint64_t Limit, const Analysis &A) {
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

class TypeInfoFI : public FunctionInfo {
  unsigned TypeInfoArgNr;

public:
  TypeInfoFI(ReturnType::Type returnType, unsigned tiArgNr)
      : FunctionInfo(returnType), TypeInfoArgNr(tiArgNr) {}

  bool analyze(LLCallBasePtr CB, const Analysis &A) override {
    Value *TypeInfo = CB->getArgOperand(TypeInfoArgNr);
    Ty = A.getTypeFor(TypeInfo);
    if (!Ty) {
      return false;
    }
    return A.DL.getTypeAllocSize(Ty) < SizeLimit;
  }
};

class ArrayFI : public TypeInfoFI {
  int ArrSizeArgNr;
  bool Initialized;
  Value *arrSize;

public:
  ArrayFI(ReturnType::Type returnType, unsigned tiArgNr, unsigned arrSizeArgNr,
          bool initialized)
      : TypeInfoFI(returnType, tiArgNr), ArrSizeArgNr(arrSizeArgNr),
        Initialized(initialized) {}

  bool analyze(LLCallBasePtr CB, const Analysis &A) override {
    if (!TypeInfoFI::analyze(CB, A)) {
      return false;
    }

    arrSize = CB->getArgOperand(ArrSizeArgNr);

    // Extract the element type from the array type.
    const StructType *ArrTy = dyn_cast<StructType>(Ty);
    assert(ArrTy && "Dynamic array type not a struct?");
    assert(isa<IntegerType>(ArrTy->getElementType(0)));
    const PointerType *PtrTy = cast<PointerType>(ArrTy->getElementType(1));
    Ty = PtrTy->getPointerElementType();

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

  Value *promote(LLCallBasePtr CB, IRBuilder<> &B, const Analysis &A) override {
    // If the allocation is of constant size it's best to put it in the
    // entry block, so do so if we're not already there.
    // For dynamically-sized allocations it's best to avoid the overhead
    // of allocating them if possible, so leave those where they are.
    // While we're at it, update statistics too.
    const IRBuilderBase::InsertPointGuard savedInsertPoint(B);
    if (isa<Constant>(arrSize)) {
      BasicBlock &Entry = CB->getCaller()->getEntryBlock();
      if (B.GetInsertBlock() != &Entry) {
        B.SetInsertPoint(&Entry, Entry.begin());
      }
      NumGcToStack++;
    } else {
      NumToDynSize++;
    }

    // Convert array size to 32 bits if necessary
    Value *count = B.CreateIntCast(arrSize, B.getInt32Ty(), false);
    AllocaInst *alloca =
        B.CreateAlloca(Ty, count, ".nongc_mem"); // FIXME: align?

    if (Initialized) {
      // For now, only zero-init is supported.
      uint64_t size = A.DL.getTypeStoreSize(Ty);
      Value *TypeSize = ConstantInt::get(arrSize->getType(), size);
      // Use the original B to put initialization at the
      // allocation site.
      Value *Size = B.CreateMul(TypeSize, arrSize);
      EmitMemZero(B, alloca, Size, A);
    }

    if (ReturnType == ReturnType::Array) {
      Value *arrStruct = llvm::UndefValue::get(CB->getType());
      arrStruct = B.CreateInsertValue(arrStruct, arrSize, 0);
      Value *memPtr =
          B.CreateBitCast(alloca, PointerType::getUnqual(B.getInt8Ty()));
      arrStruct = B.CreateInsertValue(arrStruct, memPtr, 1);
      return arrStruct;
    }

    return alloca;
  }
};

// FunctionInfo for _d_allocclass
class AllocClassFI : public FunctionInfo {
public:
  bool analyze(LLCallBasePtr CB, const Analysis &A) override {
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

  // The default promote() should be fine.

  AllocClassFI() : FunctionInfo(ReturnType::Pointer) {}
};

/// Describes runtime functions that allocate a chunk of memory with a
/// given size.
class UntypedMemoryFI : public FunctionInfo {
  unsigned SizeArgNr;
  Value *SizeArg;

public:
  bool analyze(LLCallBasePtr CB, const Analysis &A) override {
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
    Ty = CB->getType()->getContainedType(0);
    return true;
  }

  Value *promote(LLCallBasePtr CB, IRBuilder<> &B, const Analysis &A) override {
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

  explicit UntypedMemoryFI(unsigned sizeArgNr)
      : FunctionInfo(ReturnType::Pointer), SizeArgNr(sizeArgNr) {}
};
}

//===----------------------------------------------------------------------===//
// GarbageCollect2Stack Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This pass replaces GC calls with alloca's
///
class LLVM_LIBRARY_VISIBILITY GarbageCollect2Stack : public FunctionPass {
  StringMap<FunctionInfo *> KnownFunctions;
  Module *M;

  TypeInfoFI AllocMemoryT;
  ArrayFI NewArrayU;
  ArrayFI NewArrayT;
  AllocClassFI AllocClass;
  UntypedMemoryFI AllocMemory;

public:
  static char ID; // Pass identification
  GarbageCollect2Stack();

  bool doInitialization(Module &M) override {
    this->M = &M;
    return false;
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<CallGraphWrapperPass>();
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
    : FunctionPass(ID), AllocMemoryT(ReturnType::Pointer, 0),
      NewArrayU(ReturnType::Array, 0, 1, false),
      NewArrayT(ReturnType::Array, 0, 1, true), AllocMemory(0) {
  KnownFunctions["_d_allocmemoryT"] = &AllocMemoryT;
  KnownFunctions["_d_newarrayU"] = &NewArrayU;
  KnownFunctions["_d_newarrayT"] = &NewArrayT;
  KnownFunctions["_d_allocclass"] = &AllocClass;
  KnownFunctions["_d_allocmemory"] = &AllocMemory;
}

static void RemoveCall(LLCallBasePtr CB, const Analysis &A) {
  // For an invoke instruction, we insert a branch to the normal target BB
  // immediately before it. Ideally, we would find a way to not invalidate
  // the dominator tree here.
  if (auto Invoke = dyn_cast<InvokeInst>(static_cast<Instruction *>(CB))) {
    BranchInst::Create(Invoke->getNormalDest(), Invoke);
    Invoke->getUnwindDest()->removePredecessor(CB->getParent());
  }

  // Remove the runtime call.
  if (A.CGNode) {
#if LDC_LLVM_VER >= 900
    A.CGNode->removeCallEdgeFor(*CB);
#elif LDC_LLVM_VER >= 800
    A.CGNode->removeCallEdgeFor(llvm::CallSite(CB));
#else
    A.CGNode->removeCallEdgeFor(CB);
#endif
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
bool GarbageCollect2Stack::runOnFunction(Function &F) {
  LLVM_DEBUG(errs() << "\nRunning -dgc2stack on function " << F.getName() << '\n');

  const DataLayout &DL = F.getParent()->getDataLayout();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  CallGraphWrapperPass *CGPass = getAnalysisIfAvailable<CallGraphWrapperPass>();
  CallGraph *CG = CGPass ? &CGPass->getCallGraph() : nullptr;
  CallGraphNode *CGNode = CG ? (*CG)[&F] : nullptr;

  Analysis A = {DL, *M, CG, CGNode};

  BasicBlock &Entry = F.getEntryBlock();

  IRBuilder<> AllocaBuilder(&Entry, Entry.begin());

  bool Changed = false;
  for (auto &BB : F) {
    for (auto I = BB.begin(), E = BB.end(); I != E;) {
      auto originalI = I;

      // Ignore non-calls.
      Instruction *Inst = &(*(I++));
#if LDC_LLVM_VER >= 800
      auto CB = dyn_cast<CallBase>(Inst);
      if (!CB) {
        continue;
      }
#else
      LLCallBasePtr CB(Inst);
      if (!CB->getInstruction()) {
        continue;
      }
#endif

      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CB->getCalledFunction();
      if (Callee == nullptr || !Callee->isDeclaration() ||
          !Callee->hasExternalLinkage()) {
        continue;
      }

      // Ignore unknown calls.
      auto OMI = KnownFunctions.find(Callee->getName());
      if (OMI == KnownFunctions.end()) {
        continue;
      }

      FunctionInfo *info = OMI->getValue();

      if (static_cast<Instruction *>(CB)->use_empty()) {
        Changed = true;
        NumDeleted++;
        RemoveCall(CB, A);
        continue;
      }

      LLVM_DEBUG(errs() << "GarbageCollect2Stack inspecting: " << *CB);

      if (!info->analyze(CB, A)) {
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

llvm::Type *Analysis::getTypeFor(Value *typeinfo) const {
  GlobalVariable *ti_global =
      dyn_cast<GlobalVariable>(typeinfo->stripPointerCasts());
  if (!ti_global) {
    return nullptr;
  }

  const auto metaname = getMetadataName(TD_PREFIX, ti_global);

  NamedMDNode *meta = M.getNamedMetadata(metaname);
  if (!meta || meta->getNumOperands() != 1) {
    return nullptr;
  }

  MDNode *node = meta->getOperand(0);
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
#if LDC_LLVM_VER >= 800
      auto CB = llvm::cast<CallBase>(I);
#else
      LLCallBasePtr CB(I);
#endif
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
