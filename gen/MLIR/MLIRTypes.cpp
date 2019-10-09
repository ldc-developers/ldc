
//===-- irtype.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
/*
#include "gen/MLIR/MLIRTypes.h"
#include "MLIRTypes.h"
#include "mlir/IR/Builders.h"

#include "dmd/expression.h"
#include "dmd/mtype.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "mlir/IR/MLIRContext.h"

// These functions use getGlobalContext() as they are invoked before gIR
// is set.
class MLIRType{
public:

MLIRType::MLIRType(Type *dt, mlir::Type *lt, mlir::MLIRContext &mlirContext)
            : dtype(dt), type(lt), context(mlirContext), builder(&mlirContext) {
  assert(dt && "null D Type");
  assert(lt && "null MLIR Type");
  assert(!dt->ctype && "already has MLIRType");
}
private:
  /// MLIR Context
  mlir::MLIRContext &context;

  /// MLIR Builder
  mlir::OpBuilder builder;

  Type *dtype = nullptr;

  /// MLIR type.
  mlir::Type *type = nullptr;

  auto int8 = builder.getIntegerType(8);
//}

/*MLIRFuncTy &MLIRType::getMLIRFuncTy() {
  mlir_unreachable("cannot get MLIRFuncTy from non lazy/function/delegate");
}*/
/*
//////////////////////////////////////////////////////////////////////////////

MLIRTypeBasic::MLIRTypeBasic(Type *dt, mlir::MLIRContext &mlirContext)
              : MLIRType(dt, basic2mlir(dt, mlirContext),mlirContext) {}

/*MLIRTypeBasic *MLIRTypeBasic::get(Type *dt) {
  auto t = new MLIRTypeBasic(dt);
  dt->ctype = t;
  return t;
}*/

/*LLType *MLIRTypeBasic::getComplexType(mlir::MLIRContext &ctx,
                                                            mlir::Type *type) {
  mlir::Type *types[] = {type, type};
  return mlir::StructType::get(ctx, types, false);
}

namespace {
mlir::Type *getReal80Type(mlir::MLIRContext &ctx) {
  mlir::Triple::ArchType const a = global.params.targetTriple->getArch();
  bool const anyX86 = (a == mlir::Triple::x86) || (a == mlir::Triple::x86_64);
  bool const anyAarch64 =
      (a == mlir::Triple::aarch64) || (a == mlir::Triple::aarch64_be);
  bool const isAndroid =
      global.params.targetTriple->getEnvironment() == mlir::Triple::Android;

  // only x86 has 80bit float - but no support with MS C Runtime!
  if (anyX86 && !global.params.targetTriple->isWindowsMSVCEnvironment() &&
      !isAndroid) {
    return mlir::Type::getX86_FP80Ty(ctx);
  }

  if (anyAarch64 || (isAndroid && a == mlir::Triple::x86_64)) {
    return mlir::Type::getFP128Ty(ctx);
  }

  return mlir::Type::getDoubleTy(ctx);
}
}*/
/*
mlir::Type *MLIRTypeBasic::basic2mlir(Type *t, mlir::MLIRContext &ctx) {

  switch (t->ty) {
 /* case Tvoid:
    return mlir::Type void //TODO:ow to do this for MLIR?*/
/*
  case Tint8:
  case Tuns8:
  case Tchar:
    mlir::IntegerType *int8 = builder->getIntegerType(8);
    return dynamic_cast<mlir::Type*>(int8);

  case Tint16:
  case Tuns16:
  case Twchar:
    return mlir::IntegerType builder.getIntegerType(16);

  case Tint32:
  case Tuns32:
  case Tdchar:
    return mlir::IntegerType builder.getIntegerType(32);

  case Tint64:
  case Tuns64:
    return mlir::IntegerType builder.getIntegerType(64);

  case Tint128:
  case Tuns128:
    return mlir::IntegerType builder.getIntegerType(128);

 /* case Tfloat32:
  case Timaginary32:
    return mlir::Type::getFloatTy(ctx);

  case Tfloat64:
  case Timaginary64:
    return mlir::Type::getDoubleTy(ctx);

  case Tfloat80:
  case Timaginary80:
    return getReal80Type(ctx);

  case Tcomplex32:
    return getComplexType(ctx, mlir::Type::getFloatTy(ctx));

  case Tcomplex64:
    return getComplexType(ctx, mlir::Type::getDoubleTy(ctx));

  case Tcomplex80:
    return getComplexType(ctx, getReal80Type(ctx));

  case Tbool:
    return mlir::Type::getInt1Ty(ctx);*/
 /* default:
    IF_LOG Logger::println("Unknown basic type.");
    return nullptr;
  }
}

//////////////////////////////////////////////////////////////////////////////
/*
MLIRTypePointer::MLIRTypePointer(Type *dt, LLType *lt) : MLIRType(dt, lt) {}

MLIRTypePointer *MLIRTypePointer::get(Type *dt) {
  assert(!dt->ctype);
  assert((dt->ty == Tpointer || dt->ty == Tnull) && "not pointer/null type");

  LLType *elemType;
  if (dt->ty == Tnull) {
    elemType = mlir::Type::getInt8Ty(getGlobalContext());
  } else {
    elemType = DtoMemType(dt->nextOf());

    // DtoType could have already created the same type, e.g. for
    // dt == Node* in struct Node { Node* n; }.
    if (dt->ctype) {
      return dt->ctype->isPointer();
    }
  }

  auto t = new MLIRTypePointer(dt, mlir::PointerType::get(elemType, 0));
  dt->ctype = t;
  return t;
}

//////////////////////////////////////////////////////////////////////////////

MLIRTypeSArray::MLIRTypeSArray(Type *dt, LLType *lt) : MLIRType(dt, lt) {}

MLIRTypeSArray *MLIRTypeSArray::get(Type *dt) {
  assert(!dt->ctype);
  assert(dt->ty == Tsarray && "not static array type");

  LLType *elemType = DtoMemType(dt->nextOf());

  // We might have already built the type during DtoMemType e.g. as part of a
  // forward reference in a struct.
  if (!dt->ctype) {
    TypeSArray *tsa = static_cast<TypeSArray *>(dt);
    uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    dt->ctype = new MLIRTypeSArray(dt, mlir::ArrayType::get(elemType, dim));
  }

  return dt->ctype->isSArray();
}

//////////////////////////////////////////////////////////////////////////////

MLIRTypeArray::MLIRTypeArray(Type *dt, LLType *lt) : MLIRType(dt, lt) {}

MLIRTypeArray *MLIRTypeArray::get(Type *dt) {
  assert(!dt->ctype);
  assert(dt->ty == Tarray && "not dynamic array type");

  LLType *elemType = DtoMemType(dt->nextOf());

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers.
  if (!dt->ctype) {
    mlir::Type *types[] = {DtoSize_t(), mlir::PointerType::get(elemType, 0)};
    LLType *at = mlir::StructType::get(getGlobalContext(), types, false);
    dt->ctype = new MLIRTypeArray(dt, at);
  }

  return dt->ctype->isArray();
}

//////////////////////////////////////////////////////////////////////////////

MLIRTypeVector::MLIRTypeVector(Type *dt, mlir::Type *lt) : MLIRType(dt, lt) {}

MLIRTypeVector *MLIRTypeVector::get(Type *dt) {
  LLType *lt = vector2mlir(dt);
  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!dt->ctype) {
    dt->ctype = new MLIRTypeVector(dt, lt);
  }
  return dt->ctype->isVector();
}

mlir::Type *MLIRTypeVector::vector2mlir(Type *dt) {
  assert(dt->ty == Tvector && "not vector type");
  TypeVector *tv = static_cast<TypeVector *>(dt);
  assert(tv->basetype->ty == Tsarray);
  TypeSArray *tsa = static_cast<TypeSArray *>(tv->basetype);
  uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
  LLType *elemType = DtoMemType(tsa->next);
  return mlir::VectorType::get(elemType, dim);
}
}
 */
