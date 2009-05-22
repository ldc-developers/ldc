#ifndef LDC_GEN_IRSTATE_H
#define LDC_GEN_IRSTATE_H

#include <vector>
#include <deque>
#include <list>
#include <sstream>

#include "root.h"
#include "aggregate.h"

#include "ir/irfunction.h"
#include "ir/irstruct.h"
#include "ir/irvar.h"

#include "llvm/Support/CallSite.h"

namespace llvm {
    class TargetMachine;
}

// global ir state for current module
struct IRState;
struct TargetABI;

extern IRState* gIR;
extern llvm::TargetMachine* gTargetMachine;
extern const llvm::TargetData* gTargetData;
extern TargetABI* gABI;

struct TypeFunction;
struct TypeStruct;
struct ClassDeclaration;
struct FuncDeclaration;
struct Module;
struct TypeStruct;
struct BaseClass;
struct AnonDeclaration;

struct IrModule;

// represents a scope
struct IRScope
{
    llvm::BasicBlock* begin;
    llvm::BasicBlock* end;
    IRBuilder<> builder;

    IRScope();
    IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e);
};

struct IRBuilderHelper
{
    IRState* state;
    IRBuilder<>* operator->();
};

struct IRAsmStmt
{
    IRAsmStmt() 
    : isBranchToLabel(NULL) {}

    std::string code;
    std::string out_c;
    std::string in_c;
    std::vector<LLValue*> out;
    std::vector<LLValue*> in;

    // if this is nonzero, it contains the target label
    Identifier* isBranchToLabel;
};

struct IRAsmBlock
{
    std::deque<IRAsmStmt*> s;
    std::set<std::string> clobs;
    size_t outputcount;

    // stores the labels within the asm block
    std::vector<Identifier*> internalLabels;

    AsmBlockStatement* asmBlock;
    const LLType* retty;
    unsigned retn;
    bool retemu; // emulate abi ret with a temporary
    LLValue* (*retfixup)(IRBuilderHelper b, LLValue* orig); // Modifies retval

    IRAsmBlock(AsmBlockStatement* b)
        : asmBlock(b), retty(NULL), retn(0), retemu(false), retfixup(NULL),
          outputcount(0)
    {}
};

// represents the module
struct IRState
{
    IRState(llvm::Module* m);

    // module
    Module* dmodule;
    llvm::Module* module;

    // interface info type, used in DtoInterfaceInfoType
    const LLStructType* interfaceInfoType;
    const LLStructType* mutexType;
    const LLStructType* moduleRefType;

    // functions
    typedef std::vector<IrFunction*> FunctionVector;
    FunctionVector functions;
    IrFunction* func();

    llvm::Function* topfunc();
    TypeFunction* topfunctype();
    llvm::Instruction* topallocapoint();

    // structs
    typedef std::vector<IrStruct*> StructVector;
    StructVector structs;
    IrStruct* topstruct();

    // D main function
    bool emitMain;
    llvm::Function* mainFunc;

    // basic block scopes
    std::vector<IRScope> scopes;
    IRScope& scope();
    llvm::BasicBlock* scopebb();
    llvm::BasicBlock* scopeend();
    bool scopereturned();

    // create a call or invoke, depending on the landing pad info
    // the template function is defined further down in this file
    template <typename InputIterator>
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, InputIterator ArgBegin, InputIterator ArgEnd, const char* Name="");
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, const char* Name="");
    llvm::CallSite CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, const char* Name="");
    llvm::CallSite CreateCallOrInvoke2(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, const char* Name="");
    llvm::CallSite CreateCallOrInvoke3(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, LLValue* Arg3, const char* Name="");
    llvm::CallSite CreateCallOrInvoke4(LLValue* Callee, LLValue* Arg1, LLValue* Arg2,  LLValue* Arg3, LLValue* Arg4, const char* Name="");

    // this holds the array being indexed or sliced so $ will work
    // might be a better way but it works. problem is I only get a
    // VarDeclaration for __dollar, but I can't see how to get the
    // array pointer from this :(
    std::vector<DValue*> arrays;

    // builder helper
    IRBuilderHelper ir;

    // debug info helper
    llvm::DIFactory difactory;

    // static ctors/dtors/unittests
    typedef std::vector<FuncDeclaration*> FuncDeclVector;
    FuncDeclVector ctors;
    FuncDeclVector dtors;
    FuncDeclVector unitTests;

    // for inline asm
    IRAsmBlock* asmBlock;
    std::ostringstream nakedAsm;

    // 'used' array solely for keeping a reference to globals
    std::vector<LLConstant*> usedArray;

    // dwarf dbg stuff
    LLGlobalVariable* dwarfCUs;
    LLGlobalVariable* dwarfSPs;
    LLGlobalVariable* dwarfGVs;
};

template <typename InputIterator>
llvm::CallSite IRState::CreateCallOrInvoke(LLValue* Callee, InputIterator ArgBegin, InputIterator ArgEnd, const char* Name)
{
    llvm::BasicBlock* pad = func()->landingPad;
    if(pad)
    {
        // intrinsics don't support invoking and 'nounwind' functions don't need it.
        LLFunction* funcval = llvm::dyn_cast<LLFunction>(Callee);
        if (funcval && (funcval->isIntrinsic() || funcval->doesNotThrow()))
        {
            llvm::CallInst* call = ir->CreateCall(Callee, ArgBegin, ArgEnd, Name);
            call->setAttributes(funcval->getAttributes());
            return call;
        }

        llvm::BasicBlock* postinvoke = llvm::BasicBlock::Create("postinvoke", topfunc(), scopeend());
        llvm::InvokeInst* invoke = ir->CreateInvoke(Callee, postinvoke, pad, ArgBegin, ArgEnd, Name);
        if (LLFunction* fn = llvm::dyn_cast<LLFunction>(Callee))
            invoke->setAttributes(fn->getAttributes());
        scope() = IRScope(postinvoke, scopeend());
        return invoke;
    }
    else
    {
        llvm::CallInst* call = ir->CreateCall(Callee, ArgBegin, ArgEnd, Name);
        if (LLFunction* fn = llvm::dyn_cast<LLFunction>(Callee))
            call->setAttributes(fn->getAttributes());
        return call;
    }
}

#endif // LDC_GEN_IRSTATE_H
