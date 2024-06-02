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

#include "gen/abi/abi.h"
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
                                   sizeInBits / 8,
                                   /*Scalable=*/false);
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
    DtoMemSetZero(DtoType(dty), lval,
                  DtoConstSize_t(getTypeAllocSize(DtoType(dty))));
    DtoPaddedStruct(dty->toBasetype(), v, lval);
    return lval;
  }

  LLType *type(Type *t) override {
    return DtoUnpaddedStructType(t->toBasetype());
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Base for ABI rewrites bit-casting an argument to another LL type.
 * If the argument isn't in memory already, it is dumped to memory to perform
 * the bit-cast.
 */
struct BaseBitcastABIRewrite : ABIRewrite {
  static unsigned getMaxAlignment(LLType *llType, Type *dType) {
    return std::max(getABITypeAlign(llType), DtoAlignment(dType));
  }

  LLValue *put(DValue *dv, bool, bool) override {
    LLType *asType = type(dv->type);
    const unsigned alignment = getMaxAlignment(asType, dv->type);
    const char *name = ".BaseBitcastABIRewrite_arg";

    if (!dv->isLVal()) {
      LLValue *dump = DtoAllocaDump(dv, asType, alignment,
                                    ".BaseBitcastABIRewrite_arg_storage");
      return DtoLoad(asType, dump, name);
    }

    LLValue *address = DtoLVal(dv);
    LLType *pointeeType = DtoType(dv->type);

    if (getTypeStoreSize(asType) > getTypeAllocSize(pointeeType) ||
        alignment > DtoAlignment(dv->type)) {
      // not enough allocated memory or insufficiently aligned
      LLValue *paddedDump = DtoRawAlloca(
          asType, alignment, ".BaseBitcastABIRewrite_padded_arg_storage");
      DtoMemCpy(paddedDump, address,
                DtoConstSize_t(getTypeAllocSize(pointeeType)));
      return DtoLoad(asType, paddedDump, name);
    }

    return DtoLoad(asType, address, name);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    const unsigned alignment = getMaxAlignment(v->getType(), dty);
    return DtoAllocaDump(v, DtoType(dty), alignment,
                         ".BaseBitcastABIRewrite_param_storage");
  }

  void applyToIfNotObsolete(IrFuncTyArg &arg, LLType *finalLType = nullptr) {
    if (!finalLType)
      finalLType = type(arg.type);
    if (!LLTypeMemoryLayout::typesAreEquivalent(arg.ltype, finalLType))
      applyTo(arg, finalLType);
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Bit-casts an argument based on the front-end toArgTypes* machinery.
 */
struct ArgTypesRewrite : BaseBitcastABIRewrite {
  LLType *type(Type *t) override {
    LLType *rewrittenType = TargetABI::getRewrittenArgType(t->toBasetype());
    assert(rewrittenType);
    return rewrittenType;
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Bit-casts an argument to an integer of the same or next bigger size.
 */
struct IntegerRewrite : BaseBitcastABIRewrite {
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

  LLType *type(Type *t) override { return getIntegerType(t->size()); }
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
    return v;
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

//////////////////////////////////////////////////////////////////////////////

/**
 * Bit-casts a Homogeneous Floating-point/Vector Aggregate (HFVA) to an array
 * of floats/vectors.
 */
struct HFVAToArray : BaseBitcastABIRewrite {
  const int maxElements;

  HFVAToArray(int max = 4) : maxElements(max) {}

  LLType *type(Type *t) override {
    LLType *hfvaType = nullptr;
    if (TargetABI::isHFVA(t, maxElements, &hfvaType))
      return hfvaType;
    llvm_unreachable("Type t should be an HFVA");
  }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Bit-casts an argument to an array of integers of the specified size.
 */
template <int elementSize> struct CompositeToArray : BaseBitcastABIRewrite {
  LLType *type(Type *t) override {
    size_t length = (t->size() + elementSize - 1) / elementSize;
    return LLArrayType::get(LLIntegerType::get(gIR->context(), elementSize * 8),
                            length);
  }
};

// Bit-casts to an array of i32.
using CompositeToArray32 = CompositeToArray<4>;
// Bit-casts to an array of i64.
using CompositeToArray64 = CompositeToArray<8>;
