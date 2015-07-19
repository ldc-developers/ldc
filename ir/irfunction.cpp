//===-- irfunction.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
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

void FuncGen::pushToElemScope()
{
    toElemScopes.push(static_cast<unsigned>(temporariesToDestruct.size()));
}

void FuncGen::popToElemScope(bool destructTemporaries)
{
    assert(!toElemScopes.empty());

    const bool isOuterMost = (toElemScopes.size() == 1);
    if (destructTemporaries || isOuterMost)
    {
        int numInitialTemporaries = toElemScopes.back();
        assert(!isOuterMost || numInitialTemporaries == 0);
        this->destructTemporaries(numInitialTemporaries);
    }

    toElemScopes.pop();
}

void FuncGen::pushTemporaryToDestruct(VarDeclaration* vd)
{
    temporariesToDestruct.push(vd);
}

bool FuncGen::hasTemporariesToDestruct()
{
    return !temporariesToDestruct.empty();
}

VarDeclarations& FuncGen::getTemporariesToDestruct()
{
    return temporariesToDestruct;
}

void FuncGen::destructTemporaries(unsigned numToKeep)
{
    // pop one temporary after the other from the temporariesToDestruct stack
    // and evaluate its destructor expression
    // so when an exception occurs in a destructor expression, all older
    // temporaries (excl. the one which threw in its destructor) will be
    // destructed in a landing pad
    while (temporariesToDestruct.size() > numToKeep)
    {
        VarDeclaration* vd = temporariesToDestruct.pop();
        toElemDtor(vd->edtor);
    }
}

void FuncGen::destructAllTemporariesAndRestoreStack()
{
    VarDeclarations original = temporariesToDestruct;
    destructTemporaries(0);
    temporariesToDestruct = original;
}

void FuncGen::prepareToDestructAllTemporariesOnThrow(IRState* irState)
{
    class CallDestructors : public IRLandingPadCatchFinallyInfo
    {
    public:
        FuncGen& funcGen;
        CallDestructors(FuncGen& funcGen) : funcGen(funcGen) {}
        void toIR(LLValue*)
        {
            funcGen.destructAllTemporariesAndRestoreStack();
        }
    };

    CallDestructors* callDestructors = new CallDestructors(*this); // will leak

    // create landing pad
    llvm::BasicBlock* landingpadbb = llvm::BasicBlock::Create(irState->context(),
        "temporariesLandingPad", irState->topfunc(), irState->scopeend());

    // set up the landing pad
    landingPadInfo.addFinally(callDestructors);
    landingPadInfo.push(landingpadbb);
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

IrFunction *getIrFunc(FuncDeclaration *decl, bool create)
{
    if (!isIrFuncCreated(decl) && create)
    {
        assert(decl->ir.irFunc == NULL);
        decl->ir.irFunc = new IrFunction(decl);
        decl->ir.m_type = IrDsymbol::FuncType;
    }
    assert(decl->ir.irFunc != NULL);
    return decl->ir.irFunc;
}

bool isIrFuncCreated(FuncDeclaration *decl)
{
    int t = decl->ir.type();
    assert(t == IrDsymbol::FuncType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::FuncType;
}
