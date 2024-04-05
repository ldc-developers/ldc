//===-- ms-cxx-helper.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/ms-cxx-helper.h"

#include "dmd/target.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/mangling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

llvm::BasicBlock *getUnwindDest(llvm::Instruction *I) {
  if (auto II = llvm::dyn_cast<llvm::InvokeInst>(I))
    return II->getUnwindDest();
  if (auto CSI = llvm::dyn_cast<llvm::CatchSwitchInst>(I))
    return CSI->getUnwindDest();
  if (auto CRPI = llvm::dyn_cast<llvm::CleanupReturnInst>(I))
    return CRPI->getUnwindDest();
  return nullptr;
}

// return all basic blocks that are reachable from bb, but don't pass through
// ebb and don't follow unwinding target
void findSuccessors(std::vector<llvm::BasicBlock *> &blocks,
                    llvm::BasicBlock *bb, llvm::BasicBlock *ebb) {
  blocks.push_back(bb);
  if (bb != ebb) {
    assert(bb->getTerminator());
    for (size_t pos = 0; pos < blocks.size(); ++pos) {
      bb = blocks[pos];
      if (auto term = bb->getTerminator()) {
        llvm::BasicBlock *unwindDest = getUnwindDest(term);
        unsigned cnt = term->getNumSuccessors();
        for (unsigned s = 0; s < cnt; s++) {
          llvm::BasicBlock *succ = term->getSuccessor(s);
          if (succ != ebb && succ != unwindDest &&
              std::find(blocks.begin(), blocks.end(), succ) == blocks.end()) {
            blocks.push_back(succ);
          }
        }
      }
    }
    blocks.push_back(ebb);
  }
}

// remap values in all instructions of all blocks
void remapBlocks(std::vector<llvm::BasicBlock *> &blocks,
                 llvm::ValueToValueMapTy &VMap) {
  for (llvm::BasicBlock *bb : blocks)
    for (auto &I : *bb) {
      llvm::RemapInstruction(&I, VMap,
                             llvm::RF_IgnoreMissingLocals |
                                 llvm::RF_NoModuleLevelChanges);
    }
}

void remapBlocksValue(std::vector<llvm::BasicBlock *> &blocks,
                      llvm::Value *from, llvm::Value *to) {
  llvm::ValueToValueMapTy VMap;
  VMap[from] = to;
  remapBlocks(blocks, VMap);
}

// make a copy of all blocks and instructions in srcblocks
// - map values to clones
// - redirect srcTarget to continueWith
// - set "funclet" attribute inside catch/cleanup pads
// - inside funclets, replace "unreachable" with "branch cleanupret"
void cloneBlocks(const std::vector<llvm::BasicBlock *> &srcblocks,
                 std::vector<llvm::BasicBlock *> &blocks,
                 llvm::BasicBlock *continueWith, llvm::BasicBlock *unwindTo,
                 llvm::Value *funclet) {
  llvm::ValueToValueMapTy VMap;
  // map the terminal branch to the new target
  if (continueWith)
    if (auto term = srcblocks.back()->getTerminator())
      if (auto succ = term->getSuccessor(0))
        VMap[succ] = continueWith;

  for (auto bb : srcblocks) {
    llvm::Function *F = bb->getParent();

    auto nbb = llvm::BasicBlock::Create(bb->getContext(), bb->getName());
    // Loop over all instructions, and copy them over.
    for (auto &II : *bb) {
      llvm::Instruction *Inst = &II;
      llvm::Instruction *newInst = nullptr;
      if (funclet && !llvm::isa<llvm::IntrinsicInst>(Inst)) {
        if (auto IInst = llvm::dyn_cast<llvm::InvokeInst>(Inst)) {
          auto invoke = llvm::InvokeInst::Create(
              IInst, llvm::OperandBundleDef("funclet", funclet));
          newInst = invoke;
        } else if (auto CInst = llvm::dyn_cast<llvm::CallInst>(Inst)) {
          auto call = llvm::CallInst::Create(
              CInst, llvm::OperandBundleDef("funclet", funclet));
          newInst = call;
        } else if (funclet && llvm::isa<llvm::UnreachableInst>(Inst)) {
          newInst = llvm::BranchInst::Create(continueWith); // to cleanupret
        }
      }
      if (!newInst)
        newInst = Inst->clone();

#if LDC_LLVM_VER < 1600
      nbb->getInstList().push_back(newInst);
#else
      newInst->insertInto(nbb, nbb->end());
#endif

      VMap[Inst] = newInst; // Add instruction map to value.
      if (unwindTo)
        if (auto dest = getUnwindDest(Inst))
          VMap[dest] = unwindTo;
    }
    nbb->insertInto(F, continueWith);
    VMap[bb] = nbb;
    blocks.push_back(nbb);
  }

  remapBlocks(blocks, VMap);
}

bool isCatchSwitchBlock(llvm::BasicBlock *bb) {
  if (bb->empty())
    return false;
  return llvm::dyn_cast<llvm::CatchSwitchInst>(&bb->front());
}

// copy from clang/.../MicrosoftCXXABI.cpp

// routines for constructing the llvm types for MS RTTI structs.
llvm::StructType *getTypeDescriptorType(IRState &irs,
                                        llvm::Constant *classInfoPtr,
                                        llvm::StringRef TypeInfoString) {
  llvm::SmallString<256> TDTypeName("rtti.TypeDescriptor");
  TDTypeName += llvm::utostr(TypeInfoString.size());
  llvm::StructType *&TypeDescriptorType =
      irs.TypeDescriptorTypeMap[TypeInfoString.size()];
  if (TypeDescriptorType)
    return TypeDescriptorType;
  auto int8Ty = LLType::getInt8Ty(gIR->context());
  llvm::Type *FieldTypes[] = {
      classInfoPtr->getType(), // CGM.Int8PtrPtrTy,
      getPtrToType(int8Ty),    // CGM.Int8PtrTy,
      llvm::ArrayType::get(int8Ty, TypeInfoString.size() + 1)};
  TypeDescriptorType =
      llvm::StructType::create(gIR->context(), FieldTypes, TDTypeName);
  return TypeDescriptorType;
}

llvm::GlobalVariable *getTypeDescriptor(IRState &irs, ClassDeclaration *cd) {
  if (cd->isCPPclass()) {
    const char *name = target.cpp.typeInfoMangle(cd);
    return declareGlobal(cd->loc, irs.module, getVoidPtrType(), name,
                         /*isConstant*/ true, false,
                         /*useDLLImport*/ cd->isExport());
  }

  llvm::GlobalVariable *&Var = irs.TypeDescriptorMap[cd];
  if (Var)
    return Var;

  auto classInfoPtr = getIrAggr(cd, true)->getClassInfoSymbol();

  // The type name must match the expectation in druntime's ldc.eh_msvc - the
  // TypeInfo_Class name with a 'D' prefix (the first character is skipped in
  // debugger output).
  const auto TypeNameString =
      (llvm::Twine("D") + cd->toPrettyChars(/*QualifyTypes=*/true)).str();

  const auto TypeDescName = getIRMangledAggregateName(cd, "@TypeDescriptor");

  // Declare and initialize the TypeDescriptor.
  llvm::Constant *Fields[] = {
      classInfoPtr,                                     // VFPtr
      llvm::ConstantPointerNull::get(getVoidPtrType()), // Runtime data
      llvm::ConstantDataArray::getString(gIR->context(), TypeNameString)};
  llvm::StructType *TypeDescriptorType =
      getTypeDescriptorType(irs, classInfoPtr, TypeNameString);

  const LinkageWithCOMDAT lwc = {LLGlobalVariable::LinkOnceODRLinkage, true};
  Var = defineGlobal(cd->loc, gIR->module, TypeDescName,
                     llvm::ConstantStruct::get(TypeDescriptorType, Fields),
                     lwc.first, /*isConstant=*/true);
  setLinkage(lwc, Var);

  return Var;
}
