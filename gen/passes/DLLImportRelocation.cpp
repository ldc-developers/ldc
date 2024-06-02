//===-- DLLImportRelocation.cpp -------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This transform scans the initializers of global variables for references to
// dllimported globals. Such references need to be 'relocated' manually on
// Windows to prevent undefined-symbol linker errors. This is done by
// 1) nullifying the pointers in the static initializer, and
// 2) initializing these fields at runtime via a CRT constructor.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dllimport-relocation"

#include "gen/passes/Passes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

STATISTIC(NumPatchedGlobals,
          "Number of global variables with patched initializer");
STATISTIC(NumRelocations,
          "Total number of patched references to dllimported globals");

namespace {
struct LLVM_LIBRARY_VISIBILITY DLLImportRelocation {

  // Returns true if the module has been changed.
  bool run(Module &m);
};

struct LLVM_LIBRARY_VISIBILITY DLLImportRelocationLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  DLLImportRelocationLegacyPass() : ModulePass(ID) {}

  DLLImportRelocation pass;
  // Returns true if the module has been changed.
  bool runOnModule(Module &m) override { return pass.run(m); };
};

struct Impl {
  Module &m;

  // the global variable whose initializer is being fixed up
  GlobalVariable *currentGlobal = nullptr;
  // the GEP indices from the global to the currently inspected field
  SmallVector<uint64_t, 4> currentGepPath;

  Impl(Module &m) : m(m) {}

  ~Impl() {
    if (ctor) {
      // append a `ret void` instruction
      ReturnInst::Create(m.getContext(), &ctor->back());
    }
  }

  // Recursively walks over each field of an initializer and checks for
  // references to dllimported globals.
  // Returns true if a fixup was necessary.
  bool fixup(Constant *initializer) {
    if (!initializer)
      return false;

    // set i to the initializer, descending into GEPs and skipping over casts
    auto i = skipOverCast(initializer);
    if (auto gep = isGEP(i)) {
      i = skipOverCast(gep->getOperand(0));
    }

    // check if i is a reference to a dllimport global
    if (auto globalRef = dyn_cast<GlobalVariable>(i)) {
      if (globalRef->hasDLLImportStorageClass()) {
        onDLLImportReference(globalRef, initializer);
        return true;
      }
      return false;
    }

    const Type *t = initializer->getType();
    auto st = dyn_cast<StructType>(t);
    auto at = dyn_cast<ArrayType>(t);
    if (st || at) {
      // descend recursively into each field/element
      const uint64_t N = st ? st->getNumElements() : at->getNumElements();
      bool hasChanged = false;
      for (uint64_t i = 0; i < N; ++i) {
        currentGepPath.push_back(i);
        if (fixup(initializer->getAggregateElement(i)))
          hasChanged = true;
        currentGepPath.pop_back();
      }
      return hasChanged;
    }

    return false;
  }

private:
  void onDLLImportReference(GlobalVariable *importedVar,
                            Constant *originalInitializer) {
    ++NumRelocations;

    // initialize at runtime:
    currentGlobal->setConstant(false);
    appendToCRTConstructor(importedVar, originalInitializer);

    const auto pathLength = currentGepPath.size();
    if (pathLength == 0) {
      currentGlobal->setInitializer(
          Constant::getNullValue(currentGlobal->getValueType()));
      return;
    }

    // We cannot mutate a llvm::Constant, so need to replace all parent
    // aggregate initializers.
    SmallVector<Constant *, 4> initializers;
    initializers.reserve(pathLength + 1);
    initializers.push_back(currentGlobal->getInitializer());
    for (uint64_t i = 0; i < pathLength - 1; ++i) {
      initializers.push_back(
          initializers.back()->getAggregateElement(currentGepPath[i]));
    }

    // Nullify the field referencing the dllimported global.
    const auto fieldIndex = currentGepPath.back();
    auto fieldType =
        initializers.back()->getAggregateElement(fieldIndex)->getType();
    initializers.push_back(Constant::getNullValue(fieldType));

    // Replace all parent aggregate initializers, bottom-up.
    for (ptrdiff_t i = pathLength - 1; i >= 0; --i) {
      initializers[i] =
          replaceField(initializers[i], currentGepPath[i], initializers[i + 1]);
    }

    currentGlobal->setInitializer(initializers[0]);
  }

  static Constant *replaceField(Constant *aggregate, uint64_t fieldIndex,
                                Constant *newFieldValue) {
    const auto t = aggregate->getType();
    const auto st = dyn_cast<StructType>(t);
    const auto at = dyn_cast<ArrayType>(t);

    if (!st && !at) {
      llvm_unreachable("Only expecting IR structs or arrays here");
      return aggregate;
    }

    const auto N = st ? st->getNumElements() : at->getNumElements();
    std::vector<Constant *> elements;
    elements.reserve(N);
    for (uint64_t i = 0; i < N; ++i)
      elements.push_back(aggregate->getAggregateElement(i));
    elements[fieldIndex] = newFieldValue;
    return st ? ConstantStruct::get(st, elements)
              : ConstantArray::get(at, elements);
  }

  Function *ctor = nullptr;

  void createCRTConstructor() {
    ctor = Function::Create(
        FunctionType::get(Type::getVoidTy(m.getContext()), false),
        GlobalValue::PrivateLinkage, "ldc.dllimport_relocation", &m);
    llvm::BasicBlock::Create(m.getContext(), "", ctor);

    llvm::appendToGlobalCtors(m, ctor, 0);
  }

  void appendToCRTConstructor(GlobalVariable *importedVar,
                              Constant *originalInitializer) {
    if (!ctor)
      createCRTConstructor();

    IRBuilder<> b(&ctor->back());

    Value *address = currentGlobal;
    Type * t = currentGlobal->getValueType();
    for (auto i : currentGepPath) {
      if (i <= 0xFFFFFFFFu) {
        address = b.CreateConstInBoundsGEP2_32(t, address, 0,
                                               static_cast<unsigned>(i));
      } else {
        address = b.CreateConstInBoundsGEP2_64(t, address, 0, i);
      }
      if (StructType *st = dyn_cast<StructType>(t))
        t = st->getElementType(i);
      else if (ArrayType *at = dyn_cast<ArrayType>(t))
        t = at->getElementType();
      else if (dyn_cast<PointerType>(t))
        llvm_unreachable("Shouldn't be trying to GEP a pointer in initializer");
    }

    Constant *value = importedVar;
    if (auto gep = isGEP(skipOverCast(originalInitializer))) {
      Constant *newOperand =
          createConstPointerCast(importedVar, gep->getOperand(0)->getType());
      SmallVector<Constant *, 8> newOperands;
      newOperands.push_back(newOperand);
      for (unsigned i = 1, e = gep->getNumOperands(); i != e; ++i)
        newOperands.push_back(gep->getOperand(i));
      value = gep->getWithOperands(newOperands);
    }
    value = createConstPointerCast(value, t);

    // Only modify the field if the current value is still null from the static
    // initializer. This is important for multiple definitions of a (templated)
    // global, in multiple object files linked to a binary, uniqued by the
    // linker - the chosen definition might not reference dllimported globals in
    // the same fields (and might be constant altogether if it contains no
    // dllimport refs at all).
    auto ifbb = BasicBlock::Create(m.getContext(), "if", ctor);
    auto endbb = BasicBlock::Create(m.getContext(), "endif", ctor);

    auto isStillNull =
        b.CreateICmp(CmpInst::ICMP_EQ, b.CreateLoad(t, address, false),
                     Constant::getNullValue(t));
    b.CreateCondBr(isStillNull, ifbb, endbb);

    b.SetInsertPoint(ifbb);
    b.CreateStore(value, address);
    b.CreateBr(endbb);
  }

  static Constant *skipOverCast(Constant *value) {
    if (auto ce = dyn_cast<ConstantExpr>(value)) {
      if (ce->isCast())
        return ce->getOperand(0);
    }
    return value;
  }

  static ConstantExpr *isGEP(Constant *value) {
    if (auto ce = dyn_cast<ConstantExpr>(value)) {
      if (ce->getOpcode() == Instruction::GetElementPtr)
        return ce;
    }
    return nullptr;
  }

  static Constant *createConstPointerCast(Constant *value, Type *type) {
    return value->getType() == type
               ? value
               : llvm::ConstantExpr::getPointerCast(value, type);
  }
};
}

char DLLImportRelocationLegacyPass::ID = 0;
static RegisterPass<DLLImportRelocationLegacyPass>
    X("dllimport-relocation",
      "Patch references to dllimported globals in static initializers");

ModulePass *createDLLImportRelocationPass() {
  return new DLLImportRelocationLegacyPass();
}

bool DLLImportRelocation::run(Module &m) {
  Impl impl(m);
  bool hasChanged = false;

#if LDC_LLVM_VER >= 1700
  for (GlobalVariable &global : m.globals()) {
#else
  for (GlobalVariable &global : m.getGlobalList()) {
#endif
    // TODO: thread-local globals would need to be initialized in a separate TLS
    // ctor
    if (!global.hasInitializer() || global.isThreadLocal())
      continue;

    impl.currentGlobal = &global;
    impl.currentGepPath.clear();
    if (impl.fixup(global.getInitializer())) {
      ++NumPatchedGlobals;
      hasChanged = true;
    }
  }

  return hasChanged;
}
