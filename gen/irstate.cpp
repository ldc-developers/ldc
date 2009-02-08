/* DMDFE backend stubs
 * This file contains the implementations of the backend routines.
 * For dmdfe these do nothing but print a message saying the module
 * has been parsed. Substitute your own behaviors for these routimes.
 */

#include <cstdarg>

#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"
#include "statement.h"

#include "gen/irstate.h"
#include "tollvm.h"

IRState* gIR = 0;
const llvm::TargetData* gTargetData = 0;

//////////////////////////////////////////////////////////////////////////////////////////
IRScope::IRScope()
{
    begin = end = NULL;
}

IRScope::IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e)
{
    begin = b;
    end = e;
    builder.SetInsertPoint(b);
}

//////////////////////////////////////////////////////////////////////////////////////////
IRLoopScope::IRLoopScope()
{
}

IRLoopScope::IRLoopScope(Statement* s, EnclosingHandler* enclosinghandler, llvm::BasicBlock* b, llvm::BasicBlock* e, bool isSwitch)
{
    begin = b;
    end = e;
    //builder.SetInsertPoint(b);
    this->s = s;
    this->enclosinghandler = enclosinghandler;
    this->isSwitch = isSwitch;
}

//////////////////////////////////////////////////////////////////////////////////////////
IRState::IRState(llvm::Module* m)
    : module(m), difactory(*m)
{
    interfaceInfoType = NULL;
    mutexType = NULL;
    moduleRefType = NULL;

    dmodule = 0;
    emitMain = false;
    mainFunc = 0;
    ir.state = this;
    asmBlock = NULL;

    dwarfCUs = NULL;
    dwarfSPs = NULL;
    dwarfGVs = NULL;
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

TypeFunction* IRState::topfunctype()
{
    assert(!functions.empty() && "Function stack is empty!");
    return functions.back()->type;
}

llvm::Instruction* IRState::topallocapoint()
{
    assert(!functions.empty() && "AllocaPoint stack is empty!");
    return functions.back()->allocapoint;
}

IrStruct* IRState::topstruct()
{
    assert(!structs.empty() && "Struct vector is empty!");
    return structs.back();
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

CallOrInvoke* IRState::CreateCallOrInvoke(LLValue* Callee, const char* Name)
{
    LLSmallVector<LLValue*, 1> args;
    return CreateCallOrInvoke(Callee, args.begin(), args.end(), Name);
}

CallOrInvoke* IRState::CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, const char* Name)
{
    LLSmallVector<LLValue*, 1> args;
    args.push_back(Arg1);
    return CreateCallOrInvoke(Callee, args.begin(), args.end(), Name);
}

CallOrInvoke* IRState::CreateCallOrInvoke2(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, const char* Name)
{
    LLSmallVector<LLValue*, 2> args;
    args.push_back(Arg1);
    args.push_back(Arg2);
    return CreateCallOrInvoke(Callee, args.begin(), args.end(), Name);
}

CallOrInvoke* IRState::CreateCallOrInvoke3(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, LLValue* Arg3, const char* Name)
{
    LLSmallVector<LLValue*, 3> args;
    args.push_back(Arg1);
    args.push_back(Arg2);
    args.push_back(Arg3);
    return CreateCallOrInvoke(Callee, args.begin(), args.end(), Name);
}

CallOrInvoke* IRState::CreateCallOrInvoke4(LLValue* Callee, LLValue* Arg1, LLValue* Arg2,  LLValue* Arg3, LLValue* Arg4, const char* Name)
{
    LLSmallVector<LLValue*, 4> args;
    args.push_back(Arg1);
    args.push_back(Arg2);
    args.push_back(Arg3);
    args.push_back(Arg4);
    return CreateCallOrInvoke(Callee, args.begin(), args.end(), Name);
}


//////////////////////////////////////////////////////////////////////////////////////////

IRBuilder<>* IRBuilderHelper::operator->()
{
    IRBuilder<>& b = state->scope().builder;
    assert(b.GetInsertBlock() != NULL);
    return &b;
}
