//===-- irfuncty.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irfuncty.h"
#include "mtype.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

IrFuncTyArg::IrFuncTyArg(Type *t, bool bref, AttrBuilder a)
    : type(t),
      ltype(t != Type::tvoid && bref ? DtoType(t->pointerTo()) : DtoType(t)),
      attrs(std::move(a)), byref(bref) {}

bool IrFuncTyArg::isInReg() const { return attrs.contains(LLAttribute::InReg); }
bool IrFuncTyArg::isSRet() const {
  return attrs.contains(LLAttribute::StructRet);
}
bool IrFuncTyArg::isByVal() const { return attrs.contains(LLAttribute::ByVal); }

llvm::Value *IrFuncTy::putRet(DValue *dval) {
  assert(!arg_sret);

  if (ret->rewrite) {
    Logger::println("Rewrite: putRet");
    LOG_SCOPE
    return ret->rewrite->put(dval);
  }

  return dval->getRVal();
}

llvm::Value *IrFuncTy::getRet(Type *dty, LLValue *val) {
  assert(!arg_sret);

  if (ret->rewrite) {
    Logger::println("Rewrite: getRet");
    LOG_SCOPE
    return ret->rewrite->get(dty, val);
  }

  return val;
}

void IrFuncTy::getRet(Type *dty, LLValue *val, LLValue *address) {
  assert(!arg_sret);

  if (ret->rewrite) {
    Logger::println("Rewrite: getRet (getL)");
    LOG_SCOPE
    ret->rewrite->getL(dty, val, address);
    return;
  }

  DtoStoreZextI8(val, address);
}

llvm::Value *IrFuncTy::putParam(size_t idx, DValue *dval) {
  assert(idx < args.size() && "invalid putParam");
  return putParam(*args[idx], dval);
}

llvm::Value *IrFuncTy::putParam(const IrFuncTyArg &arg, DValue *dval) {
  if (arg.rewrite) {
    Logger::println("Rewrite: putParam");
    LOG_SCOPE
    return arg.rewrite->put(dval);
  }

  return dval->getRVal();
}

void IrFuncTy::getParam(Type *dty, size_t idx, LLValue *val, LLValue *address) {
  assert(idx < args.size() && "invalid getParam");

  if (args[idx]->rewrite) {
    Logger::println("Rewrite: getParam (getL)");
    LOG_SCOPE
    args[idx]->rewrite->getL(dty, val, address);
    return;
  }

  DtoStoreZextI8(val, address);
}

AttrSet IrFuncTy::getParamAttrs(bool passThisBeforeSret) {
  AttrSet newAttrs;

  int idx = 0;

// handle implicit args
#define ADD_PA(X)                                                              \
  if (X) {                                                                     \
    newAttrs.add(idx, X->attrs);                                               \
    idx++;                                                                     \
  }

  ADD_PA(ret)

  if (arg_sret && arg_this && passThisBeforeSret) {
    ADD_PA(arg_this)
    ADD_PA(arg_sret)
  } else {
    ADD_PA(arg_sret)
    ADD_PA(arg_this)
  }

  ADD_PA(arg_nest)
  ADD_PA(arg_arguments)

#undef ADD_PA

  // Set attributes on the explicit parameters.
  const size_t n = args.size();
  for (size_t k = 0; k < n; k++) {
    const size_t i = idx + (reverseParams ? (n - k - 1) : k);
    newAttrs.add(i, args[k]->attrs);
  }

  return newAttrs;
}
