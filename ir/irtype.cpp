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
  assert(!dt->ctype && "already has IrType");
}

IrFuncTy &IrType::getIrFuncTy() {
  llvm_unreachable("cannot get IrFuncTy from non lazy/function/delegate");
}

//////////////////////////////////////////////////////////////////////////////

IrTypeBasic::IrTypeBasic(Type *dt) : IrType(dt, basic2llvm(dt)) {}

IrTypeBasic *IrTypeBasic::get(Type *dt) {
  auto t = new IrTypeBasic(dt);
  dt->ctype = t;
  return t;
}

LLType *IrTypeBasic::getComplexType(llvm::LLVMContext &ctx, LLType *type) {
  llvm::Type *types[] = {type, type};
  return llvm::StructType::get(ctx, types, false);
}

namespace {
llvm::Type *getReal80Type(llvm::LLVMContext &ctx) {
  llvm::Triple::ArchType const a = global.params.targetTriple->getArch();
  bool const anyX86 = (a == llvm::Triple::x86) || (a == llvm::Triple::x86_64);
  bool const anyAarch64 =
      (a == llvm::Triple::aarch64) || (a == llvm::Triple::aarch64_be);
  bool const isAndroid =
      global.params.targetTriple->getEnvironment() == llvm::Triple::Android;

  // only x86 has 80bit float - but no support with MS C Runtime!
  if (anyX86 && !global.params.targetTriple->isWindowsMSVCEnvironment() &&
      !isAndroid) {
    return llvm::Type::getX86_FP80Ty(ctx);
  }

  if (anyAarch64 || (isAndroid && a == llvm::Triple::x86_64)) {
    return llvm::Type::getFP128Ty(ctx);
  }

  return llvm::Type::getDoubleTy(ctx);
}
}

llvm::Type *IrTypeBasic::basic2llvm(Type *t) {
  llvm::LLVMContext &ctx = getGlobalContext();

  switch (t->ty) {
  case Tvoid:
    return llvm::Type::getVoidTy(ctx);

  case Tint8:
  case Tuns8:
  case Tchar:
    return llvm::Type::getInt8Ty(ctx);

  case Tint16:
  case Tuns16:
  case Twchar:
    return llvm::Type::getInt16Ty(ctx);

  case Tint32:
  case Tuns32:
  case Tdchar:
    return llvm::Type::getInt32Ty(ctx);

  case Tint64:
  case Tuns64:
    return llvm::Type::getInt64Ty(ctx);

  case Tint128:
  case Tuns128:
    return llvm::IntegerType::get(ctx, 128);

  case Tfloat32:
  case Timaginary32:
    return llvm::Type::getFloatTy(ctx);

  case Tfloat64:
  case Timaginary64:
    return llvm::Type::getDoubleTy(ctx);

  case Tfloat80:
  case Timaginary80:
    return getReal80Type(ctx);

  case Tcomplex32:
    return getComplexType(ctx, llvm::Type::getFloatTy(ctx));

  case Tcomplex64:
    return getComplexType(ctx, llvm::Type::getDoubleTy(ctx));

  case Tcomplex80:
    return getComplexType(ctx, getReal80Type(ctx));

  case Tbool:
    return llvm::Type::getInt1Ty(ctx);
  default:
    llvm_unreachable("Unknown basic type.");
  }
}

//////////////////////////////////////////////////////////////////////////////

IrTypePointer::IrTypePointer(Type *dt, LLType *lt) : IrType(dt, lt) {}

IrTypePointer *IrTypePointer::get(Type *dt) {
  assert(!dt->ctype);
  assert((dt->ty == Tpointer || dt->ty == Tnull) && "not pointer/null type");

  LLType *elemType;
  if (dt->ty == Tnull) {
    elemType = llvm::Type::getInt8Ty(getGlobalContext());
  } else {
    elemType = DtoMemType(dt->nextOf());

    // DtoType could have already created the same type, e.g. for
    // dt == Node* in struct Node { Node* n; }.
    if (dt->ctype) {
      return dt->ctype->isPointer();
    }
  }

  auto t = new IrTypePointer(dt, llvm::PointerType::get(elemType, 0));
  dt->ctype = t;
  return t;
}

//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type *dt, LLType *lt) : IrType(dt, lt) {}

IrTypeSArray *IrTypeSArray::get(Type *dt) {
  assert(!dt->ctype);
  assert(dt->ty == Tsarray && "not static array type");

  LLType *elemType = DtoMemType(dt->nextOf());

  // We might have already built the type during DtoMemType e.g. as part of a
  // forward reference in a struct.
  if (!dt->ctype) {
    TypeSArray *tsa = static_cast<TypeSArray *>(dt);
    uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    dt->ctype = new IrTypeSArray(dt, llvm::ArrayType::get(elemType, dim));
  }

  return dt->ctype->isSArray();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeArray::IrTypeArray(Type *dt, LLType *lt) : IrType(dt, lt) {}

IrTypeArray *IrTypeArray::get(Type *dt) {
  assert(!dt->ctype);
  assert(dt->ty == Tarray && "not dynamic array type");

  LLType *elemType = DtoMemType(dt->nextOf());

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers.
  if (!dt->ctype) {
    llvm::Type *types[] = {DtoSize_t(), llvm::PointerType::get(elemType, 0)};
    LLType *at = llvm::StructType::get(getGlobalContext(), types, false);
    dt->ctype = new IrTypeArray(dt, at);
  }

  return dt->ctype->isArray();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeVector::IrTypeVector(Type *dt, llvm::Type *lt) : IrType(dt, lt) {}

IrTypeVector *IrTypeVector::get(Type *dt) {
  LLType *lt = vector2llvm(dt);
  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!dt->ctype) {
    dt->ctype = new IrTypeVector(dt, lt);
  }
  return dt->ctype->isVector();
}

llvm::Type *IrTypeVector::vector2llvm(Type *dt) {
  assert(dt->ty == Tvector && "not vector type");
  TypeVector *tv = static_cast<TypeVector *>(dt);
  assert(tv->basetype->ty == Tsarray);
  TypeSArray *tsa = static_cast<TypeSArray *>(tv->basetype);
  uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
  LLType *elemType = DtoMemType(tsa->next);
  return llvm::VectorType::get(elemType, dim);
}
