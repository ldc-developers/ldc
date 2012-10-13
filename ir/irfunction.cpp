
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/logger.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"

#include <sstream>

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrFuncTyArg::IrFuncTyArg(Type* t, bool bref, llvm::Attributes a) : type(t)
{
    ltype = t != Type::tvoid && bref ? DtoType(t->pointerTo()) : DtoType(t);
    attrs = a;
    byref = bref;
    rewrite = NULL;
}

#if LDC_LLVM_VER >= 302
bool IrFuncTyArg::isInReg() const { return attrs.hasAttribute(llvm::Attributes::InReg); }
bool IrFuncTyArg::isSRet() const  { return attrs.hasAttribute(llvm::Attributes::StructRet); }
bool IrFuncTyArg::isByVal() const { return attrs.hasAttribute(llvm::Attributes::ByVal); }
#else
bool IrFuncTyArg::isInReg() const { return (attrs & llvm::Attribute::InReg) != 0; }
bool IrFuncTyArg::isSRet() const  { return (attrs & llvm::Attribute::StructRet) != 0; }
bool IrFuncTyArg::isByVal() const { return (attrs & llvm::Attribute::ByVal) != 0; }
#endif

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

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

llvm::Value* IrFuncTy::putParam(Type* dty, int idx, DValue* val)
{
    assert(idx >= 0 && idx < args.size() && "invalid putParam");
    if (args[idx]->rewrite) {
        Logger::println("Rewrite: putParam");
        LOG_SCOPE
        return args[idx]->rewrite->put(dty, val);
    }
    return val->getRVal();
}

llvm::Value* IrFuncTy::getParam(Type* dty, int idx, DValue* val)
{
    assert(idx >= 0 && idx < args.size() && "invalid getParam");
    if (args[idx]->rewrite) {
        Logger::println("Rewrite: getParam (get)");
        LOG_SCOPE
        return args[idx]->rewrite->get(dty, val);
    }
    return val->getRVal();
}

void IrFuncTy::getParam(Type* dty, int idx, DValue* val, llvm::Value* lval)
{
    assert(idx >= 0 && idx < args.size() && "invalid getParam");

    if (args[idx]->rewrite)
    {
        Logger::println("Rewrite: getParam (getL)");
        LOG_SCOPE
        args[idx]->rewrite->getL(dty, val, lval);
        return;
    }

    DtoStore(val->getRVal(), lval);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

FuncGen::FuncGen()
{
    landingPad = NULL;
    nextUnique.push(0);
}

std::string FuncGen::getScopedLabelName(const char* ident)
{
    if(labelScopes.empty())
        return std::string(ident);

    std::string result = "__";
    for(unsigned int i = 0; i < labelScopes.size(); ++i)
        result += labelScopes[i] + "_";
    return result + ident;
}

void FuncGen::pushUniqueLabelScope(const char* name)
{
    std::ostringstream uniquename;
    uniquename << name << nextUnique.top()++;
    nextUnique.push(0);
    labelScopes.push_back(uniquename.str());
}

void FuncGen::popLabelScope()
{
    labelScopes.pop_back();
    nextUnique.pop();
}

IrFunction::IrFunction(FuncDeclaration* fd)
{
    decl = fd;

    Type* t = fd->type->toBasetype();
    assert(t->ty == Tfunction);
    type = static_cast<TypeFunction*>(t);
    func = NULL;
    allocapoint = NULL;

    queued = false;
    defined = false;

    retArg = NULL;
    thisArg = NULL;
    nestArg = NULL;

    nestedVar = NULL;
    frameType = NULL;
    depth = -1;
    nestedContextCreated = false;
    
    _arguments = NULL;
    _argptr = NULL;
}

void IrFunction::setNeverInline()
{
#if LDC_LLVM_VER >= 302
    assert(!func->getFnAttributes().hasAttribute(llvm::Attributes::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attributes::NoInline);
#else
    assert(!func->hasFnAttr(llvm::Attribute::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::NoInline);
#endif
}

void IrFunction::setAlwaysInline()
{
#if LDC_LLVM_VER >= 302
    assert(!func->getFnAttributes().hasAttribute(llvm::Attributes::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attributes::AlwaysInline);
#else
    assert(!func->hasFnAttr(llvm::Attribute::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::AlwaysInline);
#endif
}
