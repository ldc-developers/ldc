/* DMDFE backend stubs
 * This file contains the implementations of the backend routines.
 * For dmdfe these do nothing but print a message saying the module
 * has been parsed. Substitute your own behaviors for these routimes.
 */

#include "mtype.h"
#include "gen/irstate.h"

IRState* gIR = 0;
llvm::TargetData* gTargetData = 0;

//////////////////////////////////////////////////////////////////////////////////////////
IRScope::IRScope()
{
    begin = end = 0;
    returned = false;
}

IRScope::IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e)
{
    begin = b;
    end = e;
    returned = false;
}

//////////////////////////////////////////////////////////////////////////////////////////
IRState::IRState()
{
    dmodule = 0;
    module = 0;
    inLvalue = false;
    emitMain = false;
    mainFunc = 0;
}

llvm::Function* IRState::topfunc()
{
    assert(!funcs.empty() && "Function stack is empty!");
    return funcs.top();
}

TypeFunction* IRState::topfunctype()
{
    assert(!functypes.empty() && "TypeFunction stack is empty!");
    return functypes.top();
}

llvm::Instruction* IRState::topallocapoint()
{
    assert(!functypes.empty() && "AllocaPoint stack is empty!");
    return functypes.top()->llvmAllocaPoint;
}

IRStruct& IRState::topstruct()
{
    assert(!structs.empty() && "Struct vector is empty!");
    return structs.back();
}

llvm::Value* IRState::toplval()
{
    assert(!lvals.empty() && "Lval vector is empty!");
    return lvals.back();
}

IRScope& IRState::scope()
{
    assert(!scopes.empty());
    return scopes.back();
}

llvm::BasicBlock* IRState::scopebb()
{
    return scopebegin();
}
llvm::BasicBlock* IRState::scopebegin()
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
    return scope().returned;
}

//////////////////////////////////////////////////////////////////////////////////////////

IRStruct::IRStruct()
 : recty(llvm::OpaqueType::get())
{
    type = 0;
}

IRStruct::IRStruct(TypeStruct* t)
 : recty(llvm::OpaqueType::get())
{
    type = t;
}

IRStruct::~IRStruct()
{
}
