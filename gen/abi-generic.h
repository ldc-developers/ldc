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
 * Base for ABI rewrites bit-casting an argument to another LL type.
 * If the argument isn't in memory already, it is dumped to memory to perform
 * the bit-cast.
 */
struct BaseBitcastABIRewrite : ABIRewrite {
  LLValue *put(DValue *dv, bool, bool) override {
    LLValue *address = dv->isLVal() ? DtoLVal(dv) : nullptr;
    LLType *asType = type(dv->type);
    const char *name = ".BaseBitcastABIRewrite_arg";

    if (!address) {
      address = DtoAllocaDump(dv, asType, 0, ".BaseBitcastABIRewrite_arg_storage");
    }

    LLType *pointeeType = address->getType()->getPointerElementType();

    if (asType == pointeeType) {
      return DtoLoad(address, name);
    }

    if (getTypeStoreSize(asType) > getTypeAllocSize(pointeeType)) {
      // not enough allocated memory
      LLValue *paddedDump =
          DtoRawAlloca(asType, 0, ".BaseBitcastABIRewrite_padded_arg_storage");
      DtoMemCpy(paddedDump, address,
                DtoConstSize_t(getTypeAllocSize(pointeeType)));
      return DtoLoad(paddedDump, name);
    }

    address = DtoBitCast(address, getPtrToType(asType));
    return DtoLoad(address, name);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return DtoAllocaDump(v, dty, ".BaseBitcastABIRewrite_param_storage");
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
 * Rewrites any parameter to an integer of the same or next bigger size via
 * bit-casting.
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
struct HFVAToArray : BaseBitcastABIRewrite {
  const int maxFloats = 4;

  HFVAToArray(const int max = 4) : maxFloats(max) {}

  LLType *type(Type *t) override {
    LLType *rewriteType = nullptr;
    if (TargetABI::isHFVA(t, &rewriteType, maxFloats))
      return rewriteType;
    llvm_unreachable("Type t should be an HFA");
  }
};

/**
 * Rewrite a composite as array of integers.
 */
template <int elementSize> struct CompositeToArray : BaseBitcastABIRewrite {
  LLType *type(Type *t) override {
    size_t length = (t->size() + elementSize - 1) / elementSize;
    return LLArrayType::get(LLIntegerType::get(gIR->context(), elementSize * 8),
                            length);
  }
};

// Rewrite a composite as array of i32.
using CompositeToArray32 = CompositeToArray<4>;
// Rewrite a composite as array of i64.
using CompositeToArray64 = CompositeToArray<8>;

//////////////////////////////////////////////////////////////////////////////

struct RegCount {
  char gp_regs, simd_regs;

  RegCount(char gp_regs, char simd_regs)
      : gp_regs(gp_regs), simd_regs(simd_regs) {}

  explicit RegCount(LLType *ty) : gp_regs(0), simd_regs(0) {
    if (LLStructType *structTy = isaStruct(ty)) {
      for (unsigned i = 0; i < structTy->getNumElements(); ++i) {
        RegCount elementRegCount(structTy->getElementType(i));
        gp_regs += elementRegCount.gp_regs;
        simd_regs += elementRegCount.simd_regs;
      }
    } else if (LLArrayType *arrayTy = isaArray(ty)) {
      char N = static_cast<char>(arrayTy->getNumElements());
      RegCount elementRegCount(arrayTy->getElementType());
      gp_regs = N * elementRegCount.gp_regs;
      simd_regs = N * elementRegCount.simd_regs;
    } else if (ty->isIntegerTy() || ty->isPointerTy()) {
      ++gp_regs;
    } else if (ty->isFloatingPointTy() || ty->isVectorTy()) {
      // X87 reals are passed on the stack
      if (!ty->isX86_FP80Ty()) {
        ++simd_regs;
      }
    } else {
      unsigned sizeInBits = gDataLayout->getTypeSizeInBits(ty);
      IF_LOG Logger::cout() << "RegCount: assuming 1 GP register for type "
                            << *ty << " (" << sizeInBits << " bits)\n";
      assert(sizeInBits > 0 && sizeInBits <= gDataLayout->getPointerSizeInBits());
      ++gp_regs;
    }
  }

  enum SubtractionResult {
    ArgumentFitsIn,
    ArgumentWouldFitInPartially,
    ArgumentDoesntFitIn
  };

  SubtractionResult trySubtract(const IrFuncTyArg &arg) {
    const RegCount wanted(arg.ltype);

    const bool anyRegAvailable = (wanted.gp_regs > 0 && gp_regs > 0) ||
                                 (wanted.simd_regs > 0 && simd_regs > 0);
    if (!anyRegAvailable) {
      return ArgumentDoesntFitIn;
    }

    if (gp_regs < wanted.gp_regs || simd_regs < wanted.simd_regs) {
      return ArgumentWouldFitInPartially;
    }

    gp_regs -= wanted.gp_regs;
    simd_regs -= wanted.simd_regs;

    return ArgumentFitsIn;
  }
};
