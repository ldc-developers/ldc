//===-- irstate.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/irstate.h"
#include "declaration.h"
#include "mtype.h"
#include "statement.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include <cstdarg>

IRState* gIR = 0;
llvm::TargetMachine* gTargetMachine = 0;
#if LDC_LLVM_VER >= 302
const llvm::DataLayout* gDataLayout = 0;
#else
const llvm::TargetData* gDataLayout = 0;
#endif
TargetABI* gABI = 0;

//////////////////////////////////////////////////////////////////////////////////////////
IRScope::IRScope()
    : builder(gIR->context())
{
    begin = end = NULL;
}

IRScope::IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e)
    : builder(b)
{
    begin = b;
    end = e;
}

const IRScope& IRScope::operator=(const IRScope& rhs)
{
    begin = rhs.begin;
    end = rhs.end;
    builder.SetInsertPoint(begin);
    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////

IRTargetScope::IRTargetScope(
    Statement* s,
    EnclosingTryFinally* enclosinghandler,
    llvm::BasicBlock* continueTarget,
    llvm::BasicBlock* breakTarget,
    bool onlyLabeledBreak
)
{
    this->s = s;
    this->enclosinghandler = enclosinghandler;
    this->breakTarget = breakTarget;
    this->continueTarget = continueTarget;
    this->onlyLabeledBreak = onlyLabeledBreak;
}

//////////////////////////////////////////////////////////////////////////////////////////
IRState::IRState(const char *name, llvm::LLVMContext &context)
    : module(name, context), DBuilder(this)
{
    mutexType = NULL;
    moduleRefType = NULL;

    dmodule = 0;
    mainFunc = 0;
    ir.state = this;
    asmBlock = NULL;
}

IrFunction* IRState::func()
{
    assert(!functions.empty() && "Function stack is empty!");
    return functions.back();
}

llvm::Function* IRState::topfunc()
{
    assert(!functions.empty() && "Function stack is empty!");
    return functions.back()->func;
}

llvm::Instruction* IRState::topallocapoint()
{
    assert(!functions.empty() && "AllocaPoint stack is empty!");
    return functions.back()->allocapoint;
}

IRScope& IRState::scope()
{
    assert(!scopes.empty());
    return scopes.back();
}

llvm::BasicBlock* IRState::scopebb()
{
    IRScope& s = scope();
    assert(s.begin);
    return s.begin;
}
llvm::BasicBlock* IRState::scopeend()
{
    IRScope& s = scope();
    assert(s.end);
    return s.end;
}
bool IRState::scopereturned()
{
    //return scope().returned;
    return !scopebb()->empty() && scopebb()->back().isTerminator();
}

LLCallSite IRState::CreateCallOrInvoke(LLValue* Callee, const char* Name)
{
    LLSmallVector<LLValue*, 1> args;
    return CreateCallOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, const char* Name)
{
    LLValue* args[] = { Arg1 };
    return CreateCallOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke2(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, const char* Name)
{
    LLValue* args[] = { Arg1, Arg2 };
    return CreateCallOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke3(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, LLValue* Arg3, const char* Name)
{
    LLValue* args[] = { Arg1, Arg2, Arg3 };
    return CreateCallOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke4(LLValue* Callee, LLValue* Arg1, LLValue* Arg2,  LLValue* Arg3, LLValue* Arg4, const char* Name)
{
    LLValue* args[] = { Arg1, Arg2, Arg3, Arg4 };
    return CreateCallOrInvoke(Callee, args, Name);
}

bool IRState::emitArrayBoundsChecks()
{
    int p = global.params.useArrayBounds;

    // 0 or 2 are absolute decisions.
    if (p != 1) return p != 0;

    // Safe functions only.
    if (functions.empty()) return false;

    Type* t = func()->decl->type;
    return t->ty == Tfunction && ((TypeFunction*)t)->trust == TRUSTsafe;
}


//////////////////////////////////////////////////////////////////////////////////////////

IRBuilder<>* IRBuilderHelper::operator->()
{
    IRBuilder<>& b = state->scope().builder;
    assert(b.GetInsertBlock() != NULL);
    return &b;
}
