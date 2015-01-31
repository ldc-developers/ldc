//===-- irfunction.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include <sstream>

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
#if LDC_LLVM_VER >= 303
    assert(!func->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::NoInline);
#elif LDC_LLVM_VER == 302
    assert(!func->getFnAttributes().hasAttribute(llvm::Attributes::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attributes::NoInline);
#else
    assert(!func->hasFnAttr(llvm::Attribute::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::NoInline);
#endif
}

void IrFunction::setAlwaysInline()
{
#if LDC_LLVM_VER >= 303
    assert(!func->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::AlwaysInline);
#elif LDC_LLVM_VER == 302
    assert(!func->getFnAttributes().hasAttribute(llvm::Attributes::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attributes::AlwaysInline);
#else
    assert(!func->hasFnAttr(llvm::Attribute::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::AlwaysInline);
#endif
}


