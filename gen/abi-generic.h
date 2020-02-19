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

#pragma once

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
  // Vector types are folded to a universal vector type.
  static LLType *fold(LLType *type) {
    // T* => same-sized integer
    if (type->isPointerTy()) {
      return LLIntegerType::get(gIR->context(), getTypeBitSize(type));
    }

    // <N x T> => same-sized <M x i8>
    if (type->isVectorTy()) {
      const size_t sizeInBits = getTypeBitSize(type);
      assert(sizeInBits % 8 == 0);
      return llvm::VectorType::get(LLIntegerType::get(gIR->context(), 8),
                                   sizeInBits / 8);
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
  LLValue *put(DValue *v, bool, bool) override {
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

  LLValue *put(DValue *dv, bool, bool) override {
    LLValue *address = getAddressOf(dv);
    LLType *integerType = getIntegerType(dv->type->size());
    return loadFromMemory(address, integerType);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return DtoAllocaDump(v, dty, ".IntegerRewrite_dump");
  }

  LLType *type(Type *t) override { return getIntegerType(t->size()); }

  void applyToIfNotObsolete(IrFuncTyArg &arg) {
    LLType *ltype = arg.ltype;
    if (!ltype->isSized()) // e.g., opaque types
    {
      IF_LOG Logger::cout()
          << "IntegerRewrite: not rewriting non-sized type " << *ltype << '\n';
      return;
    }

    LLType *integerType = getIntegerType(getTypeStoreSize(ltype));
    if (!LLTypeMemoryLayout::typesAreEquivalent(ltype, integerType))
      applyTo(arg, integerType);
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Implements indirect high-level-by-value semantics defined like this:
 * Instead of passing a copy of the original argument directly to the callee,
 * the caller makes a bitcopy on its stack first and then passes a pointer to
 * that copy to the callee.
 * The pointer is passed as regular parameter and hence occupies either a
 * register or a function parameters stack slot.
 *
 * This differs from LLVM's byval attribute for pointer parameters.
 * The byval attribute instructs LLVM to bitcopy the IR argument pointee onto
 * the callee parameters stack. The callee's IR parameter is an implicit pointer
 * to that private copy.
 */
struct IndirectByvalRewrite : ABIRewrite {
  LLValue *put(DValue *v, bool isLValueExp, bool) override {
    // if the argument expression is an rvalue and the LL value already in
    // memory, then elide an additional copy
    if (!isLValueExp && v->isLVal())
      return DtoLVal(v);

    return DtoAllocaDump(v, ".hidden_copy_for_IndirectByvalRewrite");
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return DtoBitCast(v, DtoPtrToType(dty));
  }

  LLType *type(Type *t) override { return DtoPtrToType(t); }

  void applyTo(IrFuncTyArg &arg, LLType *finalLType = nullptr) override {
    ABIRewrite::applyTo(arg, finalLType);

    // the copy is treated as a local variable of the callee
    // hence add the NoAlias and NoCapture attributes
    auto &attrs = arg.attrs;
    attrs.clear();
    attrs.addAttribute(LLAttribute::NoAlias);
    attrs.addAttribute(LLAttribute::NoCapture);
    if (auto alignment = DtoAlignment(arg.type))
      attrs.addAlignmentAttr(alignment);
  }
};

/**
 * Rewrite Homogeneous Homogeneous Floating-point/Vector Aggregate (HFVA) as
 * array of floats/vectors.
 */
struct HFVAToArray : ABIRewrite {
  const int maxFloats = 4;

  HFVAToArray(const int max = 4) : maxFloats(max) {}

  LLValue *put(DValue *dv, bool, bool) override {
    Logger::println("rewriting HFA %s -> as array", dv->type->toChars());
    LLType *t = type(dv->type);
    return DtoLoad(DtoBitCast(DtoLVal(dv), getPtrToType(t)));
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    Logger::println("rewriting array -> as HFA %s", dty->toChars());
    return DtoAllocaDump(v, dty, ".HFAToArray_dump");
  }

  LLType *type(Type *t) override {
    LLType *rewriteType = nullptr;
    if (TargetABI::isHFVA(t, &rewriteType, maxFloats))
      return rewriteType;
    llvm_unreachable("Type t should be an HFA");
  }
};

/**
 * Rewrite a composite as array of i64.
 */
struct CompositeToArray64 : ABIRewrite {
  LLValue *put(DValue *dv, bool, bool) override {
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
  LLValue *put(DValue *dv, bool, bool) override {
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
