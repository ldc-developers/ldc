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
  LLValue *put(DValue *v) override {
    return DtoUnpaddedStruct(v->type->toBasetype(), DtoLVal(v));
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    LLValue *lval = DtoAlloca(dty, ".RemoveStructPadding_dump");
    // Make sure the padding is zero, so struct comparisons work.
    // TODO: Only do this if there's padding, and/or only initialize padding.
    DtoMemSetZero(lval, DtoConstSize_t(getTypeAllocSize(DtoType(dty))));
    DtoPaddedStruct(dty->toBasetype(), v, lval);
    return lval;
  }

  LLType *type(Type *t) override {
    return DtoUnpaddedStructType(t->toBasetype());
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Rewrites any parameter to an integer of the same or next bigger size via
 * bit-casting.
 */
struct IntegerRewrite : ABIRewrite {
  static LLType *getIntegerType(unsigned minSizeInBytes) {
    if (minSizeInBytes > 16) {
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
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
      size = 16;
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

  LLValue *put(DValue *dv) override {
    LLValue *address = getAddressOf(dv);
    LLType *integerType = getIntegerType(dv->type->size());
    return loadFromMemory(address, integerType);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return DtoAllocaDump(v, dty, ".IntegerRewrite_dump");
  }

  LLType *type(Type *t) override { return getIntegerType(t->size()); }
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

  LLValue *put(DValue *v) override {
    const unsigned align = alignment(v->type);

    if (!DtoIsInMemoryOnly(v->type)) {
      return DtoAllocaDump(DtoRVal(v), align,
                           ".ExplicitByvalRewrite_dump");
    }

    LLValue *originalPointer = DtoLVal(v);
    LLType *type = originalPointer->getType()->getPointerElementType();
    LLValue *copyForCallee =
        DtoRawAlloca(type, align, ".ExplicitByvalRewrite_dump");
    DtoMemCpy(copyForCallee, originalPointer);
    return copyForCallee;
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return DtoBitCast(v, DtoPtrToType(dty));
  }

  LLType *type(Type *t) override { return DtoPtrToType(t); }

  unsigned alignment(Type *dty) const {
    return std::max(minAlignment, DtoAlignment(dty));
  }
};

/**
 * Rewrite Homogeneous Homogeneous Floating-point Aggregate (HFA) as array of
 * float type.
 */
struct HFAToArray : ABIRewrite {
  const int maxFloats = 4;

  HFAToArray(const int max = 4) : maxFloats(max) {}

  LLValue *put(DValue *dv) override {
    Logger::println("rewriting HFA %s -> as array", dv->type->toChars());
    LLType *t = type(dv->type);
    return DtoLoad(DtoBitCast(DtoLVal(dv), getPtrToType(t)));
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    Logger::println("rewriting array -> as HFA %s", dty->toChars());
    return DtoAllocaDump(v, dty, ".HFAToArray_dump");
  }

  LLType *type(Type *t) override {
    assert(t->ty == Tstruct);
    LLType *floatArrayType = nullptr;
    if (TargetABI::isHFA((TypeStruct *)t, &floatArrayType, maxFloats))
      return floatArrayType;
    llvm_unreachable("Type t should be an HFA");
  }
};

/**
 * Rewrite a composite as array of i64.
 */
struct CompositeToArray64 : ABIRewrite {
  LLValue *put(DValue *dv) override {
    Logger::println("rewriting %s -> as i64 array", dv->type->toChars());
    LLType *t = type(dv->type);
    return DtoLoad(DtoBitCast(DtoLVal(dv), getPtrToType(t)));
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    Logger::println("rewriting i64 array -> as %s", dty->toChars());
    return DtoAllocaDump(v, dty, ".CompositeToArray64_dump");
  }

  LLType *type(Type *t) override {
    // An i64 array that will hold Type 't'
    size_t sz = (t->size() + 7) / 8;
    return LLArrayType::get(LLIntegerType::get(gIR->context(), 64), sz);
  }
};

/**
 * Rewrite a composite as array of i32.
 */
struct CompositeToArray32 : ABIRewrite {
  LLValue *put(DValue *dv) override {
    Logger::println("rewriting %s -> as i32 array", dv->type->toChars());
    LLType *t = type(dv->type);
    return DtoLoad(DtoBitCast(DtoLVal(dv), getPtrToType(t)));
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    Logger::println("rewriting i32 array -> as %s", dty->toChars());
    return DtoAllocaDump(v, dty, ".CompositeToArray32_dump");
  }

  LLType *type(Type *t) override {
    // An i32 array that will hold Type 't'
    size_t sz = (t->size() + 3) / 4;
    return LLArrayType::get(LLIntegerType::get(gIR->context(), 32), sz);
  }
};

#endif
