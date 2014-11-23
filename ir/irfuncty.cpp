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

IrFuncTyArg::IrFuncTyArg(Type* t, bool bref, const AttrBuilder& a)
    : type(t),
      ltype(t != Type::tvoid && bref ? DtoType(t->pointerTo()) : DtoType(t)),
      attrs(a), byref(bref), rewrite(0)
{
}

bool IrFuncTyArg::isInReg() const { return attrs.contains(LDC_ATTRIBUTE(InReg)); }
bool IrFuncTyArg::isSRet() const  { return attrs.contains(LDC_ATTRIBUTE(StructRet)); }
bool IrFuncTyArg::isByVal() const { return attrs.contains(LDC_ATTRIBUTE(ByVal)); }

llvm::Value* IrFuncTy::putRet(Type* dty, DValue* val)
{
    assert(!arg_sret);
    if (ret->rewrite) {
        Logger::println("Rewrite: putRet");
        LOG_SCOPE
        return ret->rewrite->put(dty, val);
    }
    return val->getRVal();
}

llvm::Value* IrFuncTy::getRet(Type* dty, DValue* val)
{
    assert(!arg_sret);
    if (ret->rewrite) {
        Logger::println("Rewrite: getRet");
        LOG_SCOPE
        return ret->rewrite->get(dty, val);
    }
    return val->getRVal();
}

llvm::Value* IrFuncTy::putParam(Type* dty, size_t idx, DValue* val)
{
    assert(idx < args.size() && "invalid putParam");
    if (args[idx]->rewrite) {
        Logger::println("Rewrite: putParam");
        LOG_SCOPE
        return args[idx]->rewrite->put(dty, val);
    }
    return val->getRVal();
}

llvm::Value* IrFuncTy::getParam(Type* dty, size_t idx, DValue* val)
{
    assert(idx < args.size() && "invalid getParam");
    if (args[idx]->rewrite) {
        Logger::println("Rewrite: getParam (get)");
        LOG_SCOPE
        return args[idx]->rewrite->get(dty, val);
    }
    return val->getRVal();
}

void IrFuncTy::getParam(Type* dty, size_t idx, DValue* val, llvm::Value* lval)
{
    assert(idx < args.size() && "invalid getParam");

    if (args[idx]->rewrite)
    {
        Logger::println("Rewrite: getParam (getL)");
        LOG_SCOPE
        args[idx]->rewrite->getL(dty, val, lval);
        return;
    }

    DtoStoreZextI8(val->getRVal(), lval);
}
