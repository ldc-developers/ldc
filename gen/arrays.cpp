//===-- arrays.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/arrays.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/init.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"

static void DtoSetArray(DValue *array, DValue *rhs);

////////////////////////////////////////////////////////////////////////////////

namespace {
LLValue *DtoSlice(LLValue *ptr, LLValue *length, LLType *elemType) {
  elemType = i1ToI8(voidToI8(elemType));
  return DtoAggrPair(length, DtoBitCast(ptr, elemType->getPointerTo()));
}
}

////////////////////////////////////////////////////////////////////////////////

LLStructType *DtoArrayType(Type *arrayTy) {
  assert(arrayTy->nextOf());
  llvm::Type *elems[] = {DtoSize_t(), DtoPtrToType(arrayTy->nextOf())};
  return llvm::StructType::get(gIR->context(), elems, false);
}

LLStructType *DtoArrayType(LLType *t) {
  llvm::Type *elems[] = {DtoSize_t(), getPtrToType(t)};
  return llvm::StructType::get(gIR->context(), elems, false);
}

////////////////////////////////////////////////////////////////////////////////

LLArrayType *DtoStaticArrayType(Type *t) {
  t = t->toBasetype();
  assert(t->ty == TY::Tsarray);
  TypeSArray *tsa = static_cast<TypeSArray *>(t);
  Type *tnext = tsa->nextOf();

  return LLArrayType::get(DtoMemType(tnext), tsa->dim->toUInteger());
}

////////////////////////////////////////////////////////////////////////////////

void DtoSetArrayToNull(DValue *v) {
  IF_LOG Logger::println("DtoSetArrayToNull");
  LOG_SCOPE;

  DtoStore(LLConstant::getNullValue(DtoType(v->type)), DtoLVal(v));
}

////////////////////////////////////////////////////////////////////////////////

static void DtoArrayInit(const Loc &loc, LLValue *ptr, LLValue *length,
                         DValue *elementValue) {
  IF_LOG Logger::println("DtoArrayInit");
  LOG_SCOPE;

  // Let's first optimize all zero/i8 initializations down to a memset.
  // This simplifies codegen later on as llvm null's have no address!
  if (!elementValue->isLVal() || !DtoIsInMemoryOnly(elementValue->type)) {
    LLValue *val = DtoRVal(elementValue);
    LLConstant *constantVal = isaConstant(val);
    bool isNullConstant = (constantVal && constantVal->isNullValue());
    if (isNullConstant || val->getType() == LLType::getInt8Ty(gIR->context())) {
      LLValue *size = length;
      size_t elementSize = getTypeAllocSize(val->getType());
      if (elementSize != 1) {
        size = gIR->ir->CreateMul(length, DtoConstSize_t(elementSize),
                                  ".arraysize");
      }
      DtoMemSet(ptr, isNullConstant ? DtoConstUbyte(0) : val, size);
      return;
    }
  }

  // create blocks
  llvm::BasicBlock *condbb = gIR->insertBB("arrayinit.cond");
  llvm::BasicBlock *bodybb = gIR->insertBBAfter(condbb, "arrayinit.body");
  llvm::BasicBlock *endbb = gIR->insertBBAfter(bodybb, "arrayinit.end");

  // initialize iterator
  LLValue *itr = DtoAllocaDump(DtoConstSize_t(0), 0, "arrayinit.itr");

  // move into the for condition block, ie. start the loop
  assert(!gIR->scopereturned());
  llvm::BranchInst::Create(condbb, gIR->scopebb());

  // replace current scope
  gIR->ir->SetInsertPoint(condbb);

  LLType *sz = DtoSize_t();
  // create the condition
  LLValue *cond_val =
      gIR->ir->CreateICmpNE(DtoLoad(sz, itr), length, "arrayinit.condition");

  // conditional branch
  assert(!gIR->scopereturned());
  llvm::BranchInst::Create(bodybb, endbb, cond_val, gIR->scopebb());

  // rewrite scope
  gIR->ir->SetInsertPoint(bodybb);

  LLValue *itr_val = DtoLoad(sz,itr);
  // assign array element value
  Type *elemty = elementValue->type->toBasetype();
  DLValue arrayelem(elemty, DtoGEP1(i1ToI8(DtoType(elemty)), ptr, itr_val, "arrayinit.arrayelem"));
  DtoAssign(loc, &arrayelem, elementValue, EXP::blit);

  // increment iterator
  DtoStore(gIR->ir->CreateAdd(itr_val, DtoConstSize_t(1), "arrayinit.new_itr"),
           itr);

  // loop
  llvm::BranchInst::Create(condbb, gIR->scopebb());

  // rewrite the scope
  gIR->ir->SetInsertPoint(endbb);
}

////////////////////////////////////////////////////////////////////////////////

static Type *DtoArrayElementType(Type *arrayType) {
  assert(arrayType->toBasetype()->nextOf());
  Type *t = arrayType->toBasetype()->nextOf()->toBasetype();
  while (t->ty == TY::Tsarray) {
    t = t->nextOf()->toBasetype();
  }
  return t;
}

////////////////////////////////////////////////////////////////////////////////

static LLValue *computeSize(LLValue *length, size_t elementSize) {
  return elementSize == 1
             ? length
             : gIR->ir->CreateMul(length, DtoConstSize_t(elementSize));
};

static void copySlice(const Loc &loc, LLValue *dstarr, LLValue *dstlen,
                      LLValue *srcarr, LLValue *srclen, size_t elementSize,
                      bool knownInBounds) {
  const bool checksEnabled =
      global.params.useAssert == CHECKENABLEon || gIR->emitArrayBoundsChecks();
  if (checksEnabled && !knownInBounds) {
    LLFunction *fn = getRuntimeFunction(loc, gIR->module, "_d_array_slice_copy");
    gIR->CreateCallOrInvoke(
        fn, {dstarr, dstlen, srcarr, srclen, DtoConstSize_t(elementSize)}, "",
        /*isNothrow=*/true);
  } else {
    // We might have dstarr == srcarr at compile time, but as long as
    // sz1 == 0 at runtime, this would probably still be legal (the C spec
    // is unclear here).
    LLValue *size = computeSize(dstlen, elementSize);
    DtoMemCpy(dstarr, srcarr, size);
  }
}

////////////////////////////////////////////////////////////////////////////////

// Determine whether t is an array of structs that need a postblit.
static bool arrayNeedsPostblit(Type *t) {
  t = DtoArrayElementType(t);
  if (t->ty == TY::Tstruct) {
    return static_cast<TypeStruct *>(t)->sym->postblit != nullptr;
  }
  return false;
}

// Does array assignment (or initialization) from another array of the same
// element type or from an appropriate single element.
void DtoArrayAssign(const Loc &loc, DValue *lhs, DValue *rhs, EXP op,
                    bool canSkipPostblit) {
  IF_LOG Logger::println("DtoArrayAssign");
  LOG_SCOPE;

  Type *t = lhs->type->toBasetype();
  Type *t2 = rhs->type->toBasetype();
  assert(t->nextOf());

  // reference assignment for dynamic array?
  if (t->ty == TY::Tarray && !lhs->isSlice()) {
    assert(t2->ty == TY::Tarray || t2->ty == TY::Tsarray);
    if (rhs->isNull()) {
      DtoSetArrayToNull(lhs);
    } else {
      DtoSetArray(lhs, rhs);
    }
    return;
  }

  // EXP::blit is generated by the frontend for (default) initialization of
  // static arrays of structs with a single element.
  const bool isConstructing = (op == EXP::construct || op == EXP::blit);

  Type *const elemType = t->nextOf()->toBasetype();
  const bool needsDestruction =
      (!isConstructing && elemType->needsDestruction());
  LLValue *realLhsPtr = DtoArrayPtr(lhs);
  LLValue *lhsPtr = DtoBitCast(realLhsPtr, getVoidPtrType());
  LLValue *lhsLength = DtoArrayLen(lhs);

  // Be careful to handle void arrays correctly when modifying this (see tests
  // for DMD issue 7493).
  // TODO: This should use AssignExp::memset.
  LLValue *realRhsArrayPtr = (t2->ty == TY::Tarray || t2->ty == TY::Tsarray)
                                 ? DtoArrayPtr(rhs)
                                 : nullptr;
  if (realRhsArrayPtr && DtoMemType(t2->nextOf()) == DtoMemType(t->nextOf())) {
    // T[]  = T[]      T[]  = T[n]
    // T[n] = T[n]     T[n] = T[]
    LLValue *rhsPtr = DtoBitCast(realRhsArrayPtr, getVoidPtrType());
    LLValue *rhsLength = DtoArrayLen(rhs);

    const bool needsPostblit = (op != EXP::blit && arrayNeedsPostblit(t) &&
                                (!canSkipPostblit || t2->ty == TY::Tarray));

    if (!needsDestruction && !needsPostblit) {
      // fast version
      const size_t elementSize = getTypeAllocSize(DtoMemType(elemType));
      if (rhs->isNull()) {
        LLValue *lhsSize = computeSize(lhsLength, elementSize);
        DtoMemSetZero(getI8Type(), lhsPtr, lhsSize);
      } else {
        bool knownInBounds =
            isConstructing || (t->ty == TY::Tsarray && t2->ty == TY::Tsarray);
        if (!knownInBounds) {
          if (auto constLhsLength = llvm::dyn_cast<LLConstantInt>(lhsLength)) {
            if (auto constRhsLength =
                    llvm::dyn_cast<LLConstantInt>(rhsLength)) {
              if (constLhsLength->getValue() == constRhsLength->getValue()) {
                knownInBounds = true;
              }
            }
          }
        }
        copySlice(loc, lhsPtr, lhsLength, rhsPtr, rhsLength, elementSize,
                  knownInBounds);
      }
    } else if (isConstructing) {
      error(
          loc,
          "ICE: array construction should have been lowered to `_d_arrayctor`");
      fatal();
    } else { // assigning
      LLValue *tmpSwap = DtoAlloca(elemType, "arrayAssign.tmpSwap");
      LLFunction *fn = getRuntimeFunction(
          loc, gIR->module,
          !canSkipPostblit ? "_d_arrayassign_l" : "_d_arrayassign_r");
      gIR->CreateCallOrInvoke(
          fn, DtoTypeInfoOf(loc, elemType), DtoSlice(rhsPtr, rhsLength, getI8Type()),
          DtoSlice(lhsPtr, lhsLength, getI8Type()), DtoBitCast(tmpSwap, getVoidPtrType()));
    }
  } else {
    // scalar rhs:
    // T[]  = T     T[n][]  = T
    // T[n] = T     T[n][m] = T
    const bool needsPostblit =
        (op != EXP::blit && !canSkipPostblit && arrayNeedsPostblit(t));

    if (!needsDestruction && !needsPostblit) {
      // fast version
      const size_t lhsElementSize =
          getTypeAllocSize(DtoMemType(lhs->type->nextOf()));
      LLType *rhsType = DtoMemType(t2);
      const size_t rhsSize = getTypeAllocSize(rhsType);
      LLValue *actualPtr = DtoBitCast(realLhsPtr, rhsType->getPointerTo());
      LLValue *actualLength = lhsLength;
      if (rhsSize != lhsElementSize) {
        LLValue *lhsSize = computeSize(lhsLength, lhsElementSize);
        actualLength =
            rhsSize == 1
                ? lhsSize
                : gIR->ir->CreateExactUDiv(lhsSize, DtoConstSize_t(rhsSize));
      }
      DtoArrayInit(loc, actualPtr, actualLength, rhs);
    } else if (isConstructing) {
      error(loc, "ICE: array construction should have been lowered to "
                 "`_d_arraysetctor`");
      fatal();
    } else {
      LLFunction *fn =
          getRuntimeFunction(loc, gIR->module, "_d_arraysetassign");
      gIR->CreateCallOrInvoke(
          fn, lhsPtr, DtoBitCast(makeLValue(loc, rhs), getVoidPtrType()),
          gIR->ir->CreateTruncOrBitCast(lhsLength,
                                        LLType::getInt32Ty(gIR->context())),
          DtoTypeInfoOf(loc, stripModifiers(t2)));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

static void DtoSetArray(DValue *array, DValue *rhs) {
  IF_LOG Logger::println("SetArray");
  LLValue *arr = DtoLVal(array);
  LLType *s = DtoType(array->type);
  assert(s);
  DtoStore(DtoArrayLen(rhs), DtoGEP(s, arr, 0u, 0));
  DtoStore(DtoArrayPtr(rhs), DtoGEP(s, arr, 0, 1));
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *DtoConstArrayInitializer(ArrayInitializer *arrinit,
                                     Type *targetType, const bool isCfile) {
  IF_LOG Logger::println("DtoConstArrayInitializer: %s | %s",
                         arrinit->toChars(), targetType->toChars());
  LOG_SCOPE;

  assert(arrinit->value.length == arrinit->index.length);

  // get base array type
  Type *arrty = targetType->toBasetype();
  size_t arrlen = arrinit->dim;

  // for statis arrays, dmd does not include any trailing default
  // initialized elements in the value/index lists
  if (arrty->ty == TY::Tsarray) {
    TypeSArray *tsa = static_cast<TypeSArray *>(arrty);
    arrlen = static_cast<size_t>(tsa->dim->toInteger());
  }

  // make sure the number of initializers is sane
  if (arrinit->index.length > arrlen || arrinit->dim > arrlen) {
    error(arrinit->loc, "too many initializers, %llu, for array[%llu]",
          static_cast<unsigned long long>(arrinit->index.length),
          static_cast<unsigned long long>(arrlen));
    fatal();
  }

  // get elem type
  Type *elemty;
  if (arrty->ty == TY::Tvector) {
    elemty = static_cast<TypeVector *>(arrty)->elementType();
  } else {
    elemty = arrty->nextOf();
  }
  LLType *llelemty = DtoMemType(elemty);

  // true if array elements differ in type, can happen with array of unions
  bool mismatch = false;

  // allocate room for initializers
  std::vector<LLConstant *> initvals(arrlen, nullptr);

  // go through each initializer, they're not sorted by index by the frontend
  size_t j = 0;
  for (size_t i = 0; i < arrinit->index.length; i++) {
    // get index
    Expression *idx = arrinit->index[i];

    // idx can be null, then it's just the next element
    if (idx) {
      j = idx->toInteger();
    }
    assert(j < arrlen);

    // get value
    Initializer *val = arrinit->value[i];
    assert(val);

    // error check from dmd
    if (initvals[j] != nullptr) {
      error(arrinit->loc, "duplicate initialization for index %llu",
            static_cast<unsigned long long>(j));
    }

    LLConstant *c = DtoConstInitializer(val->loc, elemty, val, isCfile);
    assert(c);
    if (c->getType() != llelemty) {
      mismatch = true;
    }

    initvals[j] = c;
    j++;
  }

  // die now if there was errors
  if (global.errors) {
    fatal();
  }

  // Fill out any null entries still left with default values.

  // Element default initializer. Compute lazily to be able to avoid infinite
  // recursion for types with members that are default initialized to empty
  // arrays of themselves.
  LLConstant *elemDefaultInit = nullptr;
  for (size_t i = 0; i < arrlen; i++) {
    if (initvals[i] != nullptr) {
      continue;
    }

    if (!elemDefaultInit) {
      elemDefaultInit =
          DtoConstInitializer(arrinit->loc, elemty, nullptr, isCfile);
      if (elemDefaultInit->getType() != llelemty) {
        mismatch = true;
      }
    }

    initvals[i] = elemDefaultInit;
  }

  LLConstant *constarr;
  if (mismatch) {
    constarr = LLConstantStruct::getAnon(gIR->context(),
                                         initvals); // FIXME should this pack?
  } else {
    if (arrty->ty == TY::Tvector) {
      constarr = llvm::ConstantVector::get(initvals);
    } else {
      constarr =
          LLConstantArray::get(LLArrayType::get(llelemty, arrlen), initvals);
    }
  }

  //     std::cout << "constarr: " << *constarr << std::endl;

  // if the type is a static array, we're done
  if (arrty->ty == TY::Tsarray || arrty->ty == TY::Tvector) {
    return constarr;
  }

  // we need to make a global with the data, so we have a pointer to the array
  // Important: don't make the gvar constant, since this const initializer might
  // be used as an initializer for a static T[] - where modifying contents is
  // allowed.
  auto gvar = new LLGlobalVariable(gIR->module, constarr->getType(), false,
                                   LLGlobalValue::InternalLinkage, constarr,
                                   ".constarray");

  if (arrty->ty == TY::Tpointer) {
    // we need to return pointer to the static array.
    return DtoBitCast(gvar, DtoType(arrty));
  }

  LLConstant *gep = DtoGEP(gvar->getValueType(), gvar, 0u, 0u);
  gep = llvm::ConstantExpr::getBitCast(gvar, getPtrToType(llelemty));

  return DtoConstSlice(DtoConstSize_t(arrlen), gep, arrty);
}

////////////////////////////////////////////////////////////////////////////////

Expression *indexArrayLiteral(ArrayLiteralExp *ale, unsigned idx) {
  assert(idx < ale->elements->length);
  auto e = (*ale->elements)[idx];
  if (!e) {
    return ale->basis;
  }
  return e;
}

////////////////////////////////////////////////////////////////////////////////

bool isConstLiteral(Expression *e, bool immutableType) {
  // We have to check the return value of isConst specifically for '1',
  // as SymOffExp is classified as '2' and the address of a local variable is
  // not an LLVM constant.
  //
  // Examine the ArrayLiteralExps and the StructLiteralExps element by element
  // as isConst always returns 0 on those.
  switch (e->op) {
  case EXP::arrayLiteral: {
    auto ale = static_cast<ArrayLiteralExp *>(e);

    if (!immutableType) {
      // If dynamic array: assume not constant because the array is expected to
      // be newly allocated. See GH 1924.
      Type *arrayType = ale->type->toBasetype();
      if (arrayType->ty == TY::Tarray)
        return false;
    }

    for (auto el : *ale->elements) {
      if (!isConstLiteral(el ? el : ale->basis, immutableType))
        return false;
    }
  } break;

  case EXP::structLiteral: {
    auto sle = static_cast<StructLiteralExp *>(e);
    if (sle->sd->isNested())
      return false;
    for (auto el : *sle->elements) {
      if (el && !isConstLiteral(el, immutableType))
        return false;
    }
  } break;

  // isConst also returns 0 for string literals that are obviously constant.
  case EXP::string_:
    return true;

  case EXP::symbolOffset: {
    // Note: dllimported symbols are not link-time constant.
    auto soe = static_cast<SymOffExp *>(e);
    if (VarDeclaration *vd = soe->var->isVarDeclaration()) {
       return vd->isDataseg() && !vd->isImportedSymbol();
    }
    if (FuncDeclaration *fd = soe->var->isFuncDeclaration()) {
        return !fd->isImportedSymbol();
    }
    // Assume the symbol is non-const if we can't prove it is const.
    return false;
  } break;

  default:
    if (e->isConst() != 1)
      return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

llvm::Constant *arrayLiteralToConst(IRState *p, ArrayLiteralExp *ale) {
  // Build the initializer. We have to take care as due to unions in the
  // element types (with different fields being initialized), we can end up
  // with different types for the initializer values. In this case, we
  // generate a packed struct constant instead of an array constant.
  LLType *elementType = nullptr;
  bool differentTypes = false;

  std::vector<LLConstant *> vals;
  vals.reserve(ale->elements->length);
  for (unsigned i = 0; i < ale->elements->length; ++i) {
    llvm::Constant *val = toConstElem(indexArrayLiteral(ale, i), p);
    // extend i1 to i8
    if (val->getType()->isIntegerTy(1))
      val = llvm::ConstantExpr::getZExt(val, LLType::getInt8Ty(p->context()));
    if (!elementType) {
      elementType = val->getType();
    } else {
      differentTypes |= (elementType != val->getType());
    }
    vals.push_back(val);
  }

  if (differentTypes) {
    return llvm::ConstantStruct::getAnon(vals, true);
  }

  if (!elementType) {
    assert(ale->elements->length == 0);
    elementType = DtoMemType(ale->type->toBasetype()->nextOf());
    return llvm::ConstantArray::get(LLArrayType::get(elementType, 0), vals);
  }

  llvm::ArrayType *t = llvm::ArrayType::get(elementType, ale->elements->length);
  return llvm::ConstantArray::get(t, vals);
}

////////////////////////////////////////////////////////////////////////////////

void initializeArrayLiteral(IRState *p, ArrayLiteralExp *ale,
                            LLValue *dstMem, LLType *dstType) {
  size_t elemCount = ale->elements->length;

  // Don't try to write nothing to a zero-element array, we might represent it
  // as a null pointer.
  if (elemCount == 0)
    return;

  if (isConstLiteral(ale)) {
    llvm::Constant *constarr = arrayLiteralToConst(p, ale);

    // Emit a global for longer arrays, as an inline constant is always
    // lowered to a series of movs or similar at the asm level. The
    // optimizer can still decide to promote the memcpy intrinsic, so
    // the cutoff merely affects compilation speed.
    if (elemCount <= 4) {
      DtoStore(constarr, DtoBitCast(dstMem, getPtrToType(constarr->getType())));
    } else {
      auto gvar = new llvm::GlobalVariable(gIR->module, constarr->getType(),
                                           true, LLGlobalValue::InternalLinkage,
                                           constarr, ".arrayliteral");
      gvar->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
      DtoMemCpy(dstMem, gvar,
                DtoConstSize_t(getTypeAllocSize(constarr->getType())));
    }
  } else {
    // Store the elements one by one.
    for (size_t i = 0; i < elemCount; ++i) {
      Expression *rhsExp = indexArrayLiteral(ale, i);

      LLValue *lhsPtr = DtoGEP(dstType, dstMem, 0, i, "", p->scopebb());
      DLValue lhs(rhsExp->type, DtoBitCast(lhsPtr, DtoPtrToType(rhsExp->type)));

      // try to construct it in-place
      if (!toInPlaceConstruction(&lhs, rhsExp))
        DtoAssign(ale->loc, &lhs, toElem(rhsExp), EXP::blit);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
LLConstant *DtoConstSlice(LLConstant *dim, LLConstant *ptr, Type *type) {
  LLConstant *values[2] = {dim, ptr};
  LLStructType *lltype =
      type ? isaStruct(DtoType(type))
           : LLConstantStruct::getTypeForElements(gIR->context(), values);
  return LLConstantStruct::get(lltype, values);
}

////////////////////////////////////////////////////////////////////////////////

static DSliceValue *getSlice(Type *arrayType, LLValue *array) {
  LLType *llArrayType = DtoType(arrayType);
  if (array->getType() == llArrayType)
    return new DSliceValue(arrayType, array);

  LLValue *len = DtoExtractValue(array, 0, ".len");
  LLValue *ptr = DtoExtractValue(array, 1, ".ptr");
  ptr = DtoBitCast(ptr, llArrayType->getContainedType(1));

  return new DSliceValue(arrayType, len, ptr);
}

////////////////////////////////////////////////////////////////////////////////
DSliceValue *DtoNewDynArray(const Loc &loc, Type *arrayType, DValue *dim,
                            bool defaultInit) {
  IF_LOG Logger::println("DtoNewDynArray : %s", arrayType->toChars());
  LOG_SCOPE;

  Type *eltType = arrayType->toBasetype()->nextOf();

  if (eltType->size() == 0)
    return DtoNullValue(arrayType, loc)->isSlice();

  // get runtime function
  bool zeroInit = eltType->isZeroInit();
  const char *fnname = defaultInit
                           ? (zeroInit ? "_d_newarrayT" : "_d_newarrayiT")
                           : "_d_newarrayU";
  LLFunction *fn = getRuntimeFunction(loc, gIR->module, fnname);

  // typeinfo arg
  LLValue *arrayTypeInfo = DtoTypeInfoOf(loc, arrayType);

  // dim arg
  assert(DtoType(dim->type) == DtoSize_t());
  LLValue *arrayLen = DtoRVal(dim);

  // call allocator
  LLValue *newArray =
      gIR->CreateCallOrInvoke(fn, arrayTypeInfo, arrayLen, ".gc_mem");

  // return a DSliceValue with the well-known length for better optimizability
  auto ptr =
      DtoBitCast(DtoExtractValue(newArray, 1, ".ptr"), DtoPtrToType(eltType));
  return new DSliceValue(arrayType, arrayLen, ptr);
}

////////////////////////////////////////////////////////////////////////////////
DSliceValue *DtoNewMulDimDynArray(const Loc &loc, Type *arrayType,
                                  DValue **dims, size_t ndims) {
  IF_LOG Logger::println("DtoNewMulDimDynArray : %s", arrayType->toChars());
  LOG_SCOPE;

  // get value type
  Type *vtype = arrayType->toBasetype();
  for (size_t i = 0; i < ndims; ++i) {
    vtype = vtype->nextOf();
  }

  // get runtime function
  const char *fnname =
      vtype->isZeroInit() ? "_d_newarraymTX" : "_d_newarraymiTX";
  LLFunction *fn = getRuntimeFunction(loc, gIR->module, fnname);

  // typeinfo arg
  LLValue *arrayTypeInfo = DtoTypeInfoOf(loc, arrayType);

  // Check if constant
  bool allDimsConst = true;
  for (size_t i = 0; i < ndims; ++i) {
    if (!isaConstant(DtoRVal(dims[i]))) {
      allDimsConst = false;
    }
  }

  // build dims
  LLValue *array;
  if (allDimsConst) {
    // Build constant array for dimensions
    std::vector<LLConstant *> argsdims;
    argsdims.reserve(ndims);
    for (size_t i = 0; i < ndims; ++i) {
      argsdims.push_back(isaConstant(DtoRVal(dims[i])));
    }

    llvm::Constant *dims = llvm::ConstantArray::get(
        llvm::ArrayType::get(DtoSize_t(), ndims), argsdims);
    auto gvar = new llvm::GlobalVariable(gIR->module, dims->getType(), true,
                                         LLGlobalValue::InternalLinkage, dims,
                                         ".dimsarray");
    array = llvm::ConstantExpr::getBitCast(gvar, getPtrToType(dims->getType()));
  } else {
    // Build static array for dimensions
    LLArrayType *type = LLArrayType::get(DtoSize_t(), ndims);
    array = DtoRawAlloca(type, 0, ".dimarray");
    for (size_t i = 0; i < ndims; ++i) {
      DtoStore(DtoRVal(dims[i]), DtoGEP(type, array, 0, i, ".ndim"));
    }
  }

  LLStructType *dtype = DtoArrayType(DtoSize_t());
  LLValue *darray = DtoRawAlloca(dtype, 0, ".array");
  DtoStore(DtoConstSize_t(ndims), DtoGEP(dtype, darray, 0u, 0, ".len"));
  DtoStore(DtoBitCast(array, getPtrToType(DtoSize_t())),
           DtoGEP(dtype, darray, 0, 1, ".ptr"));

  // call allocator
  LLValue *newptr =
      gIR->CreateCallOrInvoke(fn, arrayTypeInfo, DtoLoad(dtype, darray), ".gc_mem");

  IF_LOG Logger::cout() << "final ptr = " << *newptr << '\n';

  return getSlice(arrayType, newptr);
}

////////////////////////////////////////////////////////////////////////////////

DSliceValue *DtoAppendDChar(const Loc &loc, DValue *arr, Expression *exp,
                            const char *func) {
  LLValue *valueToAppend = DtoRVal(exp);

  // Prepare arguments
  LLFunction *fn = getRuntimeFunction(loc, gIR->module, func);

  // Call function (ref string x, dchar c)
  LLValue *newArray = gIR->CreateCallOrInvoke(
      fn, DtoBitCast(DtoLVal(arr), fn->getFunctionType()->getParamType(0)),
      DtoBitCast(valueToAppend, fn->getFunctionType()->getParamType(1)),
      ".appendedArray");

  return getSlice(arr->type, newArray);
}

////////////////////////////////////////////////////////////////////////////////

DSliceValue *DtoAppendDCharToString(const Loc &loc, DValue *arr,
                                    Expression *exp) {
  IF_LOG Logger::println("DtoAppendDCharToString");
  LOG_SCOPE;
  return DtoAppendDChar(loc, arr, exp, "_d_arrayappendcd");
}

////////////////////////////////////////////////////////////////////////////////

DSliceValue *DtoAppendDCharToUnicodeString(const Loc &loc, DValue *arr,
                                           Expression *exp) {
  IF_LOG Logger::println("DtoAppendDCharToUnicodeString");
  LOG_SCOPE;
  return DtoAppendDChar(loc, arr, exp, "_d_arrayappendwd");
}

////////////////////////////////////////////////////////////////////////////////
namespace {
// helper for eq and cmp
LLValue *DtoArrayEqCmp_impl(const Loc &loc, const char *func, DValue *l,
                            DValue *r, bool useti) {
  IF_LOG Logger::println("comparing arrays");
  LLFunction *fn = getRuntimeFunction(loc, gIR->module, func);
  assert(fn);

  // find common dynamic array type
  Type *commonType = l->type->toBasetype()->nextOf()->arrayOf();

  // cast static arrays to dynamic ones, this turns them into DSliceValues
  Logger::println("casting to dynamic arrays");
  l = DtoCastArray(loc, l, commonType);
  r = DtoCastArray(loc, r, commonType);

  LLSmallVector<LLValue *, 3> args;

  // get values, reinterpret cast to void[]
  args.push_back(DtoSlicePaint(DtoRVal(l),
                              DtoArrayType(LLType::getInt8Ty(gIR->context()))));
  args.push_back(DtoSlicePaint(DtoRVal(r),
                              DtoArrayType(LLType::getInt8Ty(gIR->context()))));

  // pass array typeinfo ?
  if (useti) {
    LLValue *tival = DtoTypeInfoOf(loc, l->type);
    args.push_back(DtoBitCast(tival, fn->getFunctionType()->getParamType(2)));
  }

  return gIR->CreateCallOrInvoke(fn, args);
}

/// When `true` is returned, the type can be compared using `memcmp`.
/// See `validCompareWithMemcmp`.
bool validCompareWithMemcmpType(Type *t) {
  switch (t->ty) {
  case TY::Tsarray: {
    auto *elemType = t->baseElemOf();
    return validCompareWithMemcmpType(elemType);
  }

  case TY::Tstruct:
    // TODO: Implement when structs can be compared with memcmp. Remember that
    // structs can have a user-defined opEquals, alignment padding bytes (in
    // arrays), and padding bytes.
    return false;

  case TY::Tvoid:
  case TY::Tint8:
  case TY::Tuns8:
  case TY::Tint16:
  case TY::Tuns16:
  case TY::Tint32:
  case TY::Tuns32:
  case TY::Tint64:
  case TY::Tuns64:
  case TY::Tint128:
  case TY::Tuns128:
  case TY::Tbool:
  case TY::Tchar:
  case TY::Twchar:
  case TY::Tdchar:
  case TY::Tpointer:
    return true;

    // TODO: Determine whether this can be "return true" too:
    // case TY::Tvector:

  default:
    return false;
  }
}

/// When `true` is returned, `l` and `r` can be compared using `memcmp`.
///
/// This function may return `false` even though `memcmp` would be valid.
/// It may only return `true` if it is 100% certain.
///
/// Comparing with memcmp is often not valid, for example due to
/// - Floating point types
/// - Padding bytes
/// - User-defined opEquals
bool validCompareWithMemcmp(DValue *l, DValue *r) {
  auto *lElemType = l->type->toBasetype()->nextOf()->toBasetype();
  auto *rElemType = r->type->toBasetype()->nextOf()->toBasetype();

  // Only memcmp equivalent element types (memcmp should be used for
  // `const int[3] == int[]`, but not for `int[3] == short[3]`).
  if (!lElemType->equivalent(rElemType))
    return false;

  return validCompareWithMemcmpType(lElemType);
}

// Create a call instruction to memcmp.
llvm::CallInst *callMemcmp(const Loc &loc, IRState &irs, LLValue *l_ptr,
                           LLValue *r_ptr, LLValue *numElements, LLType *elemty) {
  assert(l_ptr && r_ptr && numElements);
  LLFunction *fn = getRuntimeFunction(loc, gIR->module, "memcmp");
  assert(fn);
  auto sizeInBytes = numElements;
  size_t elementSize = getTypeAllocSize(elemty);
  if (elementSize != 1) {
    sizeInBytes = irs.ir->CreateMul(sizeInBytes, DtoConstSize_t(elementSize));
  }
  // Call memcmp.
  LLValue *args[] = {DtoBitCast(l_ptr, getVoidPtrType()),
                     DtoBitCast(r_ptr, getVoidPtrType()), sizeInBytes};
  return irs.ir->CreateCall(fn, args);
}

/// Compare `l` and `r` using memcmp. No checks are done for validity.
///
/// This function can deal with comparisons of static and dynamic arrays
/// with memcmp.
///
/// Note: the dynamic array length check is not covered by (LDC's) PGO.
LLValue *DtoArrayEqCmp_memcmp(const Loc &loc, DValue *l, DValue *r,
                              IRState &irs) {
  IF_LOG Logger::println("Comparing arrays using memcmp");

  auto *l_ptr = DtoArrayPtr(l);
  auto *r_ptr = DtoArrayPtr(r);
  auto *l_length = DtoArrayLen(l);

  // Early return for the simple case of comparing two static arrays.
  const bool staticArrayComparison =
      (l->type->toBasetype()->ty == TY::Tsarray) &&
      (r->type->toBasetype()->ty == TY::Tsarray);
  if (staticArrayComparison) {
    // TODO: simply codegen when comparing static arrays with different length (int[3] == int[2])
    return callMemcmp(loc, irs, l_ptr, r_ptr, l_length, DtoMemType(l->type->nextOf()));
  }

  // First compare the array lengths
  auto lengthsCompareEqual =
      irs.ir->CreateICmp(llvm::ICmpInst::ICMP_EQ, l_length, DtoArrayLen(r));

  llvm::BasicBlock *incomingBB = irs.scopebb();
  llvm::BasicBlock *memcmpBB = irs.insertBB("domemcmp");
  llvm::BasicBlock *memcmpEndBB = irs.insertBBAfter(memcmpBB, "memcmpend");
  irs.ir->CreateCondBr(lengthsCompareEqual, memcmpBB, memcmpEndBB);

  // If lengths are equal: call memcmp.
  // Note: no extra null checks are needed before passing the pointers to memcmp.
  // The array comparison is UB for non-zero length, and memcmp will correctly
  // return 0 (equality) when the length is zero.
  irs.ir->SetInsertPoint(memcmpBB);
  auto memcmpAnswer = callMemcmp(loc, irs, l_ptr, r_ptr, l_length, DtoMemType(l->type->nextOf()));
  irs.ir->CreateBr(memcmpEndBB);

  // Merge the result of length check and memcmp call into a phi node.
  irs.ir->SetInsertPoint(memcmpEndBB);
  llvm::PHINode *phi =
      irs.ir->CreatePHI(LLType::getInt32Ty(gIR->context()), 2, "cmp_result");
  phi->addIncoming(DtoConstInt(1), incomingBB);
  phi->addIncoming(memcmpAnswer, memcmpBB);

  return phi;
}
} // end anonymous namespace

////////////////////////////////////////////////////////////////////////////////
LLValue *DtoArrayEquals(const Loc &loc, EXP op, DValue *l, DValue *r) {
  LLValue *res = nullptr;

  if (r->isNull()) {
    // optimize comparisons against null by rewriting to `l.length op 0`
    const auto predicate = eqTokToICmpPred(op);
    res = gIR->ir->CreateICmp(predicate, DtoArrayLen(l), DtoConstSize_t(0));
  } else if (validCompareWithMemcmp(l, r)) {
    // Use memcmp directly if possible. This avoids typeinfo lookup, and enables
    // further optimization because LLVM understands the semantics of C's
    // `memcmp`.
    const auto predicate = eqTokToICmpPred(op);
    const auto memcmp_result = DtoArrayEqCmp_memcmp(loc, l, r, *gIR);
    res = gIR->ir->CreateICmp(predicate, memcmp_result, DtoConstInt(0));
  } else {
    res = DtoArrayEqCmp_impl(loc, "_adEq2", l, r, true);
    const auto predicate = eqTokToICmpPred(op, /* invert = */ true);
    res = gIR->ir->CreateICmp(predicate, res, DtoConstInt(0));
  }

  return res;
}

////////////////////////////////////////////////////////////////////////////////
LLValue *DtoDynArrayIs(EXP op, DValue *l, DValue *r) {
  assert(l);
  assert(r);

  LLValue *len1 = DtoArrayLen(l);
  LLValue *ptr1 = DtoArrayPtr(l);

  LLValue *len2 = DtoArrayLen(r);
  LLValue *ptr2 = DtoArrayPtr(r);

  return createIPairCmp(op, len1, ptr1, len2, ptr2);
}

////////////////////////////////////////////////////////////////////////////////
LLValue *DtoArrayLen(DValue *v) {
  IF_LOG Logger::println("DtoArrayLen");
  LOG_SCOPE;

  Type *t = v->type->toBasetype();
  if (t->ty == TY::Tarray) {
    if (v->isNull()) {
      return DtoConstSize_t(0);
    }
    if (v->isLVal()) {
      return DtoLoad(DtoSize_t(),
                     DtoGEP(DtoType(v->type), DtoLVal(v), 0u, 0), ".len");
    }
    auto slice = v->isSlice();
    assert(slice);
    return slice->getLength();
  }
  if (t->ty == TY::Tsarray) {
    assert(!v->isSlice());
    assert(!v->isNull());
    TypeSArray *sarray = static_cast<TypeSArray *>(t);
    return DtoConstSize_t(sarray->dim->toUInteger());
  }
  llvm_unreachable("unsupported array for len");
}

////////////////////////////////////////////////////////////////////////////////
LLValue *DtoArrayPtr(DValue *v) {
  IF_LOG Logger::println("DtoArrayPtr");
  LOG_SCOPE;

  Type *t = v->type->toBasetype();
  // v's LL array element type may not be the real one
  // due to implicit casts (e.g., to base class)
  LLType *wantedLLPtrType = DtoPtrToType(t->nextOf());
  LLValue *ptr = nullptr;

  if (t->ty == TY::Tarray) {
    if (v->isNull()) {
      ptr = getNullPtr(wantedLLPtrType);
    } else if (v->isLVal()) {
      ptr = DtoLoad(wantedLLPtrType, DtoGEP(DtoType(v->type), DtoLVal(v), 0, 1), ".ptr");
    } else {
      auto slice = v->isSlice();
      assert(slice);
      ptr = slice->getPtr();
    }
  } else if (t->ty == TY::Tsarray) {
    assert(!v->isSlice());
    assert(!v->isNull());
    ptr = DtoLVal(v);
  } else {
    llvm_unreachable("Unexpected array type.");
  }

  return DtoBitCast(ptr, wantedLLPtrType);
}

////////////////////////////////////////////////////////////////////////////////
DValue *DtoCastArray(const Loc &loc, DValue *u, Type *to) {
  IF_LOG Logger::println("DtoCastArray");
  LOG_SCOPE;

  LLType *tolltype = DtoType(to);

  Type *totype = to->toBasetype();
  Type *fromtype = u->type->toBasetype();
  if (fromtype->ty != TY::Tarray && fromtype->ty != TY::Tsarray) {
    error(loc, "can't cast `%s` to `%s`", u->type->toChars(), to->toChars());
    fatal();
  }

  IF_LOG Logger::cout() << "from array or sarray" << '\n';

  if (totype->ty == TY::Tpointer) {
    IF_LOG Logger::cout() << "to pointer" << '\n';
    LLValue *ptr = DtoArrayPtr(u);
    if (ptr->getType() != tolltype) {
      ptr = gIR->ir->CreateBitCast(ptr, tolltype);
    }
    return new DImValue(to, ptr);
  }

  if (totype->ty == TY::Tarray) {
    IF_LOG Logger::cout() << "to array" << '\n';

    LLValue *length = nullptr;
    LLValue *ptr = nullptr;
    if (fromtype->ty == TY::Tsarray) {
      length = DtoConstSize_t(
          static_cast<TypeSArray *>(fromtype)->dim->toUInteger());
      ptr = DtoLVal(u);
    } else {
      length = DtoArrayLen(u);
      ptr = DtoArrayPtr(u);
    }

    const auto fsize = fromtype->nextOf()->size();
    const auto tsize = totype->nextOf()->size();
    if (fsize != tsize) {
      if (auto constLength = isaConstantInt(length)) {
        // compute new constant length: (constLength * fsize) / tsize
        const auto totalSize = constLength->getZExtValue() * fsize;
        if (totalSize % tsize != 0) {
          error(loc,
                "invalid cast from `%s` to `%s`, the element sizes don't "
                "line up",
                fromtype->toChars(), totype->toChars());
          fatal();
        }
        length = DtoConstSize_t(totalSize / tsize);
      } else if (fsize % tsize == 0) {
        // compute new dynamic length: length * (fsize / tsize)
        length = gIR->ir->CreateMul(length, DtoConstSize_t(fsize / tsize));
      } else {
        llvm_unreachable("should have been lowered to `__ArrayCast`");
      }
    }

    LLType *ptrty = tolltype->getStructElementType(1);
    return new DSliceValue(to, length, DtoBitCast(ptr, ptrty));
  }

  if (totype->ty == TY::Tsarray) {
    IF_LOG Logger::cout() << "to sarray" << '\n';

    LLValue *ptr = nullptr;
    if (fromtype->ty == TY::Tsarray) {
      ptr = DtoLVal(u);
    } else {
      size_t tosize = static_cast<TypeSArray *>(totype)->dim->toInteger();
      size_t i =
          (tosize * totype->nextOf()->size() - 1) / fromtype->nextOf()->size();
      DConstValue index(Type::tsize_t, DtoConstSize_t(i));
      DtoIndexBoundsCheck(loc, u, &index);
      ptr = DtoArrayPtr(u);
    }

    return new DLValue(to, DtoBitCast(ptr, getPtrToType(tolltype)));
  }

  if (totype->ty == TY::Tbool) {
    // return (arr.ptr !is null)
    LLValue *ptr = DtoArrayPtr(u);
    LLConstant *nul = getNullPtr(ptr->getType());
    return new DImValue(to, gIR->ir->CreateICmpNE(ptr, nul));
  }

  const auto castedPtr = DtoBitCast(DtoArrayPtr(u), getPtrToType(tolltype));
  return new DLValue(to, castedPtr);
}

void DtoIndexBoundsCheck(const Loc &loc, DValue *arr, DValue *index) {
  Type *arrty = arr->type->toBasetype();
  assert((arrty->ty == TY::Tsarray || arrty->ty == TY::Tarray ||
          arrty->ty == TY::Tpointer) &&
         "Can only array bounds check for static or dynamic arrays");

  if (!index) {
    // Caller supplied no index, known in-bounds.
    return;
  }

  if (arrty->ty == TY::Tpointer) {
    // Length of pointers is unknown, ignore.
    return;
  }
  if (auto ts = arrty->isTypeSArray()) {
    if (ts->isIncomplete()) // importC
      return;
  }

  LLValue *const llIndex = DtoRVal(index);
  LLValue *const llLength = DtoArrayLen(arr);
  LLValue *const cond = gIR->ir->CreateICmp(llvm::ICmpInst::ICMP_ULT, llIndex,
                                            llLength, "bounds.cmp");

  llvm::BasicBlock *okbb = gIR->insertBB("bounds.ok");
  llvm::BasicBlock *failbb = gIR->insertBBAfter(okbb, "bounds.fail");
  gIR->ir->CreateCondBr(cond, okbb, failbb);

  // set up failbb to call the array bounds error runtime function
  gIR->ir->SetInsertPoint(failbb);
  emitArrayIndexError(gIR, loc, llIndex, llLength);

  // if ok, proceed in okbb
  gIR->ir->SetInsertPoint(okbb);
}

static void emitRangeErrorImpl(IRState *irs, const Loc &loc,
                               const char *cAssertMsg, const char *dFnName,
                               llvm::ArrayRef<LLValue *> extraArgs) {
  Module *const module = irs->func()->decl->getModule();

  switch (global.params.checkAction) {
  case CHECKACTION_C:
    DtoCAssert(module, loc, DtoConstCString(cAssertMsg));
    break;
  case CHECKACTION_halt:
    irs->ir->CreateCall(GET_INTRINSIC_DECL(trap), {});
    irs->ir->CreateUnreachable();
    break;
  case CHECKACTION_context:
  case CHECKACTION_D: {
    auto fn = getRuntimeFunction(loc, irs->module, dFnName);
    LLSmallVector<LLValue *, 5> args;
    args.reserve(2 + extraArgs.size());
    args.push_back(DtoModuleFileName(module, loc));
    args.push_back(DtoConstUint(loc.linnum()));
    args.insert(args.end(), extraArgs.begin(), extraArgs.end());
    irs->CreateCallOrInvoke(fn, args);
    irs->ir->CreateUnreachable();
    break;
  }
  default:
    llvm_unreachable("Unhandled checkAction");
  }
}

void emitRangeError(IRState *irs, const Loc &loc) {
  emitRangeErrorImpl(irs, loc, "array overflow", "_d_arraybounds", {});
}

void emitArraySliceError(IRState *irs, const Loc &loc, LLValue *lower,
                         LLValue *upper, LLValue *length) {
  emitRangeErrorImpl(irs, loc, "array slice out of bounds",
                     "_d_arraybounds_slice", {lower, upper, length});
}

void emitArrayIndexError(IRState *irs, const Loc &loc, LLValue *index,
                         LLValue *length) {
  emitRangeErrorImpl(irs, loc, "array index out of bounds",
                     "_d_arraybounds_index", {index, length});
}
