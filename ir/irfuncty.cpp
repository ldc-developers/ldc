//===-- irfuncty.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irfuncty.h"

#include "dmd/mtype.h"
#include "gen/abi/abi.h"
#include "gen/dvalue.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

using namespace dmd;

IrFuncTyArg::IrFuncTyArg(Type *t, bool bref)
    : type(t),
      ltype(t != Type::tvoid && bref ? DtoType(pointerTo(t)) : DtoType(t)),
      attrs(getGlobalContext()), byref(bref) {
  mem.addRange(&type, sizeof(type));
}

IrFuncTyArg::IrFuncTyArg(Type *t, bool bref, llvm::AttrBuilder a)
    : type(t),
      ltype(t != Type::tvoid && bref ? DtoType(pointerTo(t)) : DtoType(t)),
      attrs(std::move(a)), byref(bref) {

  mem.addRange(&type, sizeof(type));
}

IrFuncTyArg::~IrFuncTyArg() { mem.removeRange(&type); }

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
    // Choosing isLValueExp=true here is a fail-safe conservative choice.
    // Most rewrites don't care, and those which do are usually not applied to
    // the return value (as more complex types are returned via sret and so not
    // rewritten at all).
    return ret->rewrite->put(dval, /*isLValueExp=*/true,
                             /*isLastArgExp=*/true);
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

llvm::Value *IrFuncTy::putArg(const IrFuncTyArg &arg, DValue *dval,
                              bool isLValueExp, bool isLastArgExp) {
  if (arg.rewrite) {
    Logger::println("Rewrite: putArg (%s expression%s)",
                    isLValueExp ? "lvalue" : "rvalue",
                    isLastArgExp ? ", last argument" : "");
    LOG_SCOPE
    return arg.rewrite->put(dval, isLValueExp, isLastArgExp);
  }

  if (arg.byref || DtoIsInMemoryOnly(dval->type)) {
    if (isLValueExp && !isLastArgExp && arg.isByVal()) {
      // copy to avoid visibility of potential side effects of later argument
      // expressions
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
    newAttrs.addToParam(idx + k, args[k]->attrs);
  }

  return newAttrs;
}
