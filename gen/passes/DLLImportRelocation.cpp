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
#if LDC_LLVM_VER < 700
#define LLVM_DEBUG DEBUG
#endif

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
struct LLVM_LIBRARY_VISIBILITY DLLImportRelocation : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  DLLImportRelocation() : ModulePass(ID) {}

  // Returns true if the module has been changed.
  bool runOnModule(Module &m) override;
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
      ReturnInst::Create(m.getContext(), &ctor->getEntryBlock());
    }
  }

  // Recursively walks over each field of an initializer and checks for
  // references to dllimported globals.
  // Returns true if a fixup was necessary.
  bool fixup(Constant *initializer) {
    if (!initializer)
      return false;

    // set i to the initializer, skipping over an optional cast
    auto i = initializer;
    if (auto ce = dyn_cast<ConstantExpr>(i)) {
      if (ce->isCast())
        i = ce->getOperand(0);
    }

    // check if i is a reference to a dllimport global
    if (auto globalRef = dyn_cast<GlobalVariable>(i)) {
      if (globalRef->hasDLLImportStorageClass()) {
        onDLLImportReference(globalRef);
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
  void onDLLImportReference(GlobalVariable *importedVar) {
    ++NumRelocations;

    // initialize at runtime:
    currentGlobal->setConstant(false);
    appendToCRTConstructor(importedVar);

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

  void appendToCRTConstructor(GlobalVariable *importedVar) {
    if (!ctor)
      createCRTConstructor();

    IRBuilder<> b(&ctor->getEntryBlock());

    Value *address = currentGlobal;
    for (auto i : currentGepPath) {
      if (i <= 0xFFFFFFFFu) {
        address = b.CreateConstInBoundsGEP2_32(
            address->getType()->getPointerElementType(), address, 0,
            static_cast<unsigned>(i));
      } else {
        address = b.CreateConstInBoundsGEP2_64(address, 0, i);
      }
    }

    Value *value = importedVar;
    auto t = address->getType()->getPointerElementType();
    if (value->getType() != t)
      value = b.CreatePointerCast(value, t);

    b.CreateStore(value, address);
  }
};
}

char DLLImportRelocation::ID = 0;
static RegisterPass<DLLImportRelocation>
    X("dllimport-relocation",
      "Patch references to dllimported globals in static initializers");

ModulePass *createDLLImportRelocationPass() {
  return new DLLImportRelocation();
}

bool DLLImportRelocation::runOnModule(Module &m) {
  Impl impl(m);
  bool hasChanged = false;

  for (GlobalVariable &global : m.getGlobalList()) {
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
