//===-- gen/abi-generic.h - Generic Target ABI helpers ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains helpers for handling rewrites common to more than one target ABI.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_GENERIC_H
#define LDC_GEN_ABI_GENERIC_H

#include "gen/abi.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/tollvm.h"

struct LLTypeMemoryLayout {
  // Structs and static arrays are folded recursively to scalars or anonymous
  // structs.
  // Pointer types are folded to an integer type.
  static LLType *fold(LLType *type) {
    // T* => integer
    if (type->isPointerTy()) {
      return LLIntegerType::get(gIR->context(), getTypeBitSize(type));
    }

    if (LLStructType *structType = isaStruct(type)) {
      unsigned numElements = structType->getNumElements();

      // fold each element
      std::vector<LLType *> elements;
      elements.reserve(numElements);
      for (unsigned i = 0; i < numElements; ++i) {
        elements.push_back(fold(structType->getElementType(i)));
      }

      // single element? then discard wrapping struct
      if (numElements == 1) {
        return elements[0];
      }

      return LLStructType::get(gIR->context(), elements,
                               structType->isPacked());
    }

    if (LLArrayType *arrayType = isaArray(type)) {
      unsigned numElements = arrayType->getNumElements();
      LLType *foldedElementType = fold(arrayType->getElementType());

      // single element? then fold to scalar
      if (numElements == 1) {
        return foldedElementType;
      }

      // otherwise: convert to struct of N folded elements
      std::vector<LLType *> elements(numElements, foldedElementType);
      return LLStructType::get(gIR->context(), elements);
    }

    return type;
  }

  // Checks two LLVM types for memory-layout equivalency.
  static bool typesAreEquivalent(LLType *a, LLType *b) {
    if (a == b) {
      return true;
    }
    if (!a || !b) {
      return false;
    }

    return fold(a) == fold(b);
  }
};

//////////////////////////////////////////////////////////////////////////////

/// Removes padding fields for (non-union-containing!) structs
struct RemoveStructPadding : ABIRewrite {
  /// get a rewritten value back to its original form
  LLValue *get(Type *dty, LLValue *v) override {
    LLValue *lval = DtoAlloca(dty, ".rewritetmp");
    getL(dty, v, lval);
    return lval;
  }

  /// get a rewritten value back to its original form and store result in
  /// provided lvalue
  void getL(Type *dty, LLValue *v, LLValue *lval) override {
    // Make sure the padding is zero, so struct comparisons work.
    // TODO: Only do this if there's padding, and/or only initialize padding.
    DtoMemSetZero(lval, DtoConstSize_t(getTypeAllocSize(DtoType(dty))));
    DtoPaddedStruct(dty->toBasetype(), v, lval);
  }

  /// put out rewritten value
  LLValue *put(DValue *v) override {
    return DtoUnpaddedStruct(v->getType()->toBasetype(), v->getRVal());
  }

  /// return the transformed type for this rewrite
  LLType *type(Type *dty, LLType *t) override {
    return DtoUnpaddedStructType(dty->toBasetype());
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Rewrites any parameter to an integer of the same or next bigger size via
 * bit-casting.
 */
struct IntegerRewrite : ABIRewrite {
  static LLType *getIntegerType(unsigned minSizeInBytes) {
    if (minSizeInBytes > 8) {
      return nullptr;
    }

    unsigned size = minSizeInBytes;
    switch (minSizeInBytes) {
    case 0:
      size = 1;
      break;
    case 3:
      size = 4;
      break;
    case 5:
    case 6:
    case 7:
      size = 8;
      break;
    default:
      break;
    }

    return LLIntegerType::get(gIR->context(), size * 8);
  }

  static bool isObsoleteFor(LLType *llType) {
    if (!llType->isSized()) // e.g., opaque types
    {
      IF_LOG Logger::cout() << "IntegerRewrite: not rewriting non-sized type "
                            << *llType << '\n';
      return true;
    }

    LLType *integerType = getIntegerType(getTypeStoreSize(llType));
    return LLTypeMemoryLayout::typesAreEquivalent(llType, integerType);
  }

  LLValue *get(Type *dty, LLValue *v) override {
    LLValue *integerDump = DtoAllocaDump(v, dty, ".IntegerRewrite_dump");
    LLType *type = DtoType(dty);
    return loadFromMemory(integerDump, type, ".IntegerRewrite_getResult");
  }

  void getL(Type *dty, LLValue *v, LLValue *lval) override {
    storeToMemory(v, lval);
  }

  LLValue *put(DValue *dv) override {
    LLValue *address = getAddressOf(dv);
    LLType *integerType = getIntegerType(dv->getType()->size());
    return loadFromMemory(address, integerType, ".IntegerRewrite_putResult");
  }

  LLType *type(Type *t, LLType *) override { return getIntegerType(t->size()); }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Implements explicit ByVal semantics defined like this:
 * Instead of passing a copy of the original argument directly to the callee,
 * the caller makes a bitcopy on its stack first and then passes a pointer to
 * that copy to the callee.
 * The pointer is passed as regular parameter and hence occupies either a
 * register or a function parameters stack slot.
 *
 * This differs from LLVM's ByVal attribute for pointer parameters.
 * The ByVal attribute instructs LLVM to pass the pointed-to argument directly
 * as a copy on the function parameters stack. In this case, there's no need to
 * pass an explicit pointer; the address is implicit.
 */
struct ExplicitByvalRewrite : ABIRewrite {
  const unsigned minAlignment;

  explicit ExplicitByvalRewrite(unsigned minAlignment = 16)
      : minAlignment(minAlignment) {}

  LLValue *get(Type *dty, LLValue *v) override {
    return DtoLoad(v, ".ExplicitByvalRewrite_getResult");
  }

  void getL(Type *dty, LLValue *v, LLValue *lval) override {
    DtoMemCpy(lval, v);
  }

  LLValue *put(DValue *v) override {
    Type *dty = v->getType();
    const unsigned align = alignment(dty);

    if (DtoIsInMemoryOnly(dty)) {
      LLValue *originalPointer = v->getRVal();
      LLType *type = originalPointer->getType()->getPointerElementType();
      LLValue *copyForCallee =
          DtoRawAlloca(type, align, ".ExplicitByvalRewrite_putResult");
      DtoMemCpy(copyForCallee, originalPointer);
      return copyForCallee;
    }

    return DtoAllocaDump(v->getRVal(), align,
                         ".ExplicitByvalRewrite_putResult");
  }

  LLType *type(Type *dty, LLType *t) override { return DtoPtrToType(dty); }

  unsigned alignment(Type *dty) const {
    return std::max(minAlignment, DtoAlignment(dty));
  }
};

#endif
