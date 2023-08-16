//===-- irtype.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irtype.h"

#include "dmd/expression.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"

// These functions use getGlobalContext() as they are invoked before gIR
// is set.

IrType::IrType(Type *dt, LLType *lt) : dtype(dt), type(lt) {
  assert(dt && "null D Type");
  assert(lt && "null LLVM Type");
  assert(!getIrType(dt) && "already has IrType");
}

IrFuncTy &IrType::getIrFuncTy() {
  llvm_unreachable("cannot get IrFuncTy from non lazy/function/delegate");
}

//////////////////////////////////////////////////////////////////////////////

IrTypeBasic::IrTypeBasic(Type *dt) : IrType(dt, basic2llvm(dt)) {}

IrTypeBasic *IrTypeBasic::get(Type *dt) {
  auto t = new IrTypeBasic(dt);
  getIrType(dt) = t;
  return t;
}

LLType *IrTypeBasic::getComplexType(llvm::LLVMContext &ctx, LLType *type) {
  llvm::Type *types[] = {type, type};
  return llvm::StructType::get(ctx, types, false);
}

llvm::Type *IrTypeBasic::basic2llvm(Type *t) {
  llvm::LLVMContext &ctx = getGlobalContext();

  switch (t->ty) {
  case TY::Tvoid:
  case TY::Tnoreturn:
    return llvm::Type::getVoidTy(ctx);

  case TY::Tint8:
  case TY::Tuns8:
  case TY::Tchar:
    return llvm::Type::getInt8Ty(ctx);

  case TY::Tint16:
  case TY::Tuns16:
  case TY::Twchar:
    return llvm::Type::getInt16Ty(ctx);

  case TY::Tint32:
  case TY::Tuns32:
  case TY::Tdchar:
    return llvm::Type::getInt32Ty(ctx);

  case TY::Tint64:
  case TY::Tuns64:
    return llvm::Type::getInt64Ty(ctx);

  case TY::Tint128:
  case TY::Tuns128:
    return llvm::IntegerType::get(ctx, 128);

  case TY::Tfloat32:
  case TY::Timaginary32:
    return llvm::Type::getFloatTy(ctx);

  case TY::Tfloat64:
  case TY::Timaginary64:
    return llvm::Type::getDoubleTy(ctx);

  case TY::Tfloat80:
  case TY::Timaginary80:
    return target.realType;

  case TY::Tcomplex32:
    return getComplexType(ctx, llvm::Type::getFloatTy(ctx));

  case TY::Tcomplex64:
    return getComplexType(ctx, llvm::Type::getDoubleTy(ctx));

  case TY::Tcomplex80:
    return getComplexType(ctx, target.realType);

  case TY::Tbool:
    return llvm::Type::getInt1Ty(ctx);
  default:
    llvm_unreachable("Unknown basic type.");
  }
}

//////////////////////////////////////////////////////////////////////////////

IrTypePointer::IrTypePointer(Type *dt, LLType *lt) : IrType(dt, lt) {}

IrTypePointer *IrTypePointer::get(Type *dt) {
  assert((dt->ty == TY::Tpointer || dt->ty == TY::Tnull) &&
         "not pointer/null type");

  auto &ctype = getIrType(dt);
  assert(!ctype);

  LLType *elemType;
  unsigned addressSpace = 0;
  if (dt->ty == TY::Tnull) {
    elemType = llvm::Type::getInt8Ty(getGlobalContext());
  } else {
    elemType = DtoMemType(dt->nextOf());
    if (dt->nextOf()->ty == TY::Tfunction) {
      addressSpace = gDataLayout->getProgramAddressSpace();
    }

    // DtoType could have already created the same type, e.g. for
    // dt == Node* in struct Node { Node* n; }.
    if (ctype) {
      return ctype->isPointer();
    }
  }

  auto t =
      new IrTypePointer(dt, llvm::PointerType::get(elemType, addressSpace));
  ctype = t;
  return t;
}

//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type *dt, LLType *lt) : IrType(dt, lt) {}

IrTypeSArray *IrTypeSArray::get(Type *dt) {
  assert(dt->ty == TY::Tsarray && "not static array type");

  auto &ctype = getIrType(dt);
  assert(!ctype);

  LLType *elemType = DtoMemType(dt->nextOf());

  // We might have already built the type during DtoMemType e.g. as part of a
  // forward reference in a struct.
  if (!ctype) {
    TypeSArray *tsa = static_cast<TypeSArray *>(dt);
    uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    ctype = new IrTypeSArray(dt, llvm::ArrayType::get(elemType, dim));
  }

  return ctype->isSArray();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeArray::IrTypeArray(Type *dt, LLType *lt) : IrType(dt, lt) {}

IrTypeArray *IrTypeArray::get(Type *dt) {
  assert(dt->ty == TY::Tarray && "not dynamic array type");

  auto &ctype = getIrType(dt);
  assert(!ctype);

  LLType *elemType = DtoMemType(dt->nextOf());

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers.
  if (!ctype) {
    llvm::Type *types[] = {DtoSize_t(), llvm::PointerType::get(elemType, 0)};
    LLType *at = llvm::StructType::get(getGlobalContext(), types, false);
    ctype = new IrTypeArray(dt, at);
  }

  return ctype->isArray();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeVector::IrTypeVector(Type *dt, llvm::Type *lt) : IrType(dt, lt) {}

IrTypeVector *IrTypeVector::get(Type *dt) {
  TypeVector *tv = dt->isTypeVector();
  assert(tv && "not vector type");

  auto &ctype = getIrType(dt);
  assert(!ctype);

  TypeSArray *tsa = tv->basetype->isTypeSArray();
  assert(tsa);
  LLType *elemType = DtoMemType(tsa->next);

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!ctype) {
    LLType *lt = llvm::VectorType::get(elemType, tsa->dim->toUInteger(),
                                       /*Scalable=*/false);
    ctype = new IrTypeVector(dt, lt);
  }

  return ctype->isVector();
}

//////////////////////////////////////////////////////////////////////////////

IrType *&getIrType(Type *t, bool create) {
  // See remark in DtoType().
  assert(
      (t->ty != TY::Tstruct || t == static_cast<TypeStruct *>(t)->sym->type) &&
      "use sd->type for structs");
  assert((t->ty != TY::Tclass || t == static_cast<TypeClass *>(t)->sym->type) &&
         "use cd->type for classes");

  t = stripModifiers(t);

  if (create) {
    DtoType(t);
    assert(t->ctype);
  }

  return t->ctype;
}
