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
#include "gen/llvmhelpers.h"
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
    return ret->rewrite->put(dval, /*isModifiableLvalue=*/false);
  }

  if (ret->byref || DtoIsInMemoryOnly(dval->type))
    return DtoLVal(dval);

  return DtoRVal(dval);
}

llvm::Value *IrFuncTy::getRetRVal(Type *dty, LLValue *val) {
  assert(!arg_sret);

  if (ret->rewrite) {
    Logger::println("Rewrite: getRetRVal");
    LOG_SCOPE
    return ret->rewrite->getRVal(dty, val);
  }

  return val;
}

llvm::Value *IrFuncTy::getRetLVal(Type *dty, LLValue *val) {
  assert(!arg_sret);

  if (ret->rewrite) {
    Logger::println("Rewrite: getRetLVal");
    LOG_SCOPE
    return ret->rewrite->getLVal(dty, val);
  }

  return DtoAllocaDump(val, dty);
}

llvm::Value *IrFuncTy::putParam(const IrFuncTyArg &arg, DValue *dval,
                                bool isModifiableLvalue) {
  if (arg.rewrite) {
    Logger::println("Rewrite: putParam");
    LOG_SCOPE
    return arg.rewrite->put(dval, isModifiableLvalue);
  }

  if (arg.byref || DtoIsInMemoryOnly(dval->type)) {
    if (isModifiableLvalue && arg.isByVal()) {
      return DtoAllocaDump(dval, ".lval_copy_for_byval");
    }
    return DtoLVal(dval);
  }

  return DtoRVal(dval);
}

LLValue *IrFuncTy::getParamLVal(Type *dty, size_t idx, LLValue *val) {
  assert(idx < args.size() && "invalid getParam");

  if (args[idx]->rewrite) {
    Logger::println("Rewrite: getParamLVal");
    LOG_SCOPE
    return args[idx]->rewrite->getLVal(dty, val);
  }

  return DtoAllocaDump(val, dty);
}

AttrSet IrFuncTy::getParamAttrs(bool passThisBeforeSret) {
  AttrSet newAttrs;

  if (ret) {
    newAttrs.addToReturn(ret->attrs);
  }

  int idx = 0;

// handle implicit args
#define ADD_PA(X)                                                              \
  if (X) {                                                                     \
    newAttrs.addToParam(idx, (X)->attrs);                                      \
    idx++;                                                                     \
  }

  if (arg_sret && arg_this && passThisBeforeSret) {
    ADD_PA(arg_this)
    ADD_PA(arg_sret)
  } else {
    ADD_PA(arg_sret)
    ADD_PA(arg_this)
  }

  ADD_PA(arg_nest)
  ADD_PA(arg_objcSelector)
  ADD_PA(arg_arguments)

#undef ADD_PA

  // Set attributes on the explicit parameters.
  const size_t n = args.size();
  for (size_t k = 0; k < n; k++) {
    const size_t i = idx + (reverseParams ? (n - k - 1) : k);
    newAttrs.addToParam(i, args[k]->attrs);
  }

  return newAttrs;
}
