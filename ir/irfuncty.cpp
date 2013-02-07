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

#if LDC_LLVM_VER >= 303
IrFuncTyArg::IrFuncTyArg(Type* t, bool bref, llvm::AttrBuilder a)
#else
IrFuncTyArg::IrFuncTyArg(Type* t, bool bref, llvm::Attributes a)
#endif
    : type(t),
      ltype(t != Type::tvoid && bref ? DtoType(t->pointerTo()) : DtoType(t)),
      attrs(a), byref(bref), rewrite(0)
{
}

#if LDC_LLVM_VER >= 303
bool IrFuncTyArg::isInReg() const { return attrs.contains(llvm::Attribute::InReg); }
bool IrFuncTyArg::isSRet() const  { return attrs.contains(llvm::Attribute::StructRet); }
bool IrFuncTyArg::isByVal() const { return attrs.contains(llvm::Attribute::ByVal); }
#elif LDC_LLVM_VER == 302
bool IrFuncTyArg::isInReg() const { return attrs.hasAttribute(llvm::Attributes::InReg); }
bool IrFuncTyArg::isSRet() const  { return attrs.hasAttribute(llvm::Attributes::StructRet); }
bool IrFuncTyArg::isByVal() const { return attrs.hasAttribute(llvm::Attributes::ByVal); }
#else
bool IrFuncTyArg::isInReg() const { return (attrs & llvm::Attribute::InReg) != 0; }
bool IrFuncTyArg::isSRet() const  { return (attrs & llvm::Attribute::StructRet) != 0; }
bool IrFuncTyArg::isByVal() const { return (attrs & llvm::Attribute::ByVal) != 0; }
#endif

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

    DtoStore(val->getRVal(), lval);
}
