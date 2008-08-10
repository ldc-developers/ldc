#ifndef LLVMDC_GEN_IRSTATE_H
#define LLVMDC_GEN_IRSTATE_H

#include <vector>
#include <list>

#include "root.h"
#include "aggregate.h"

#include "ir/irfunction.h"
#include "ir/irstruct.h"
#include "ir/irvar.h"

// global ir state for current module
struct IRState;
extern IRState* gIR;
extern const llvm::TargetData* gTargetData;

struct TypeFunction;
struct TypeStruct;
struct ClassDeclaration;
struct FuncDeclaration;
struct Module;
struct TypeStruct;
struct BaseClass;
struct EnclosingHandler;

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

// scope for loops
struct IRLoopScope : IRScope
{
    // generating statement
    Statement* s;
    // the try of a TryFinally that encloses the loop
    EnclosingHandler* enclosinghandler;
    // if it is a switch, we are a possible target for break
    // but not for continue
    bool isSwitch;

    IRLoopScope();
    IRLoopScope(Statement* s, EnclosingHandler* enclosinghandler, llvm::BasicBlock* b, llvm::BasicBlock* e, bool isSwitch = false);
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
    std::vector<IRAsmStmt*> s;
    std::set<std::string> clobs;

    // stores the labels within the asm block
    std::vector<Identifier*> internalLabels;
};

// llvm::CallInst and llvm::InvokeInst don't share a common base
// but share common functionality. to avoid duplicating code for
// adjusting these common properties these structs are made
struct CallOrInvoke
{
    virtual void setParamAttrs(const llvm::PAListPtr& Attrs) = 0;
    virtual void setCallingConv(unsigned CC) = 0;
    virtual llvm::Instruction* get() = 0;
};

struct CallOrInvoke_Call : public CallOrInvoke
{
    llvm::CallInst* inst;
    CallOrInvoke_Call(llvm::CallInst* call) : inst(call) {}

    virtual void setParamAttrs(const llvm::PAListPtr& Attrs)
    { inst->setParamAttrs(Attrs); }
    virtual void setCallingConv(unsigned CC)
    { inst->setCallingConv(CC); }
    virtual llvm::Instruction* get()
    { return inst; }
};

struct CallOrInvoke_Invoke : public CallOrInvoke
{
    llvm::InvokeInst* inst;
    CallOrInvoke_Invoke(llvm::InvokeInst* invoke) : inst(invoke) {}

    virtual void setParamAttrs(const llvm::PAListPtr& Attrs)
    { inst->setParamAttrs(Attrs); }
    virtual void setCallingConv(unsigned CC)
    { inst->setCallingConv(CC); }
    virtual llvm::Instruction* get()
    { return inst; }
};

// represents the module
struct IRState
{
    IRState();

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

    // classes TODO move into IRClass
    typedef std::vector<ClassDeclaration*> ClassDeclVec;
    ClassDeclVec classes;

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
    CallOrInvoke* CreateCallOrInvoke(LLValue* Callee, InputIterator ArgBegin, InputIterator ArgEnd, const char* Name="");
    CallOrInvoke* CreateCallOrInvoke(LLValue* Callee, const char* Name="");
    CallOrInvoke* CreateCallOrInvoke(LLValue* Callee, LLValue* Arg1, const char* Name="");
    CallOrInvoke* CreateCallOrInvoke2(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, const char* Name="");
    CallOrInvoke* CreateCallOrInvoke3(LLValue* Callee, LLValue* Arg1, LLValue* Arg2, LLValue* Arg3, const char* Name="");
    CallOrInvoke* CreateCallOrInvoke4(LLValue* Callee, LLValue* Arg1, LLValue* Arg2,  LLValue* Arg3, LLValue* Arg4, const char* Name="");

    // loop blocks
    typedef std::vector<IRLoopScope> LoopScopeVec;
    LoopScopeVec loopbbs;

    // this holds the array being indexed or sliced so $ will work
    // might be a better way but it works. problem is I only get a
    // VarDeclaration for __dollar, but I can't see how to get the
    // array pointer from this :(
    std::vector<DValue*> arrays;

    // builder helper
    IRBuilderHelper ir;

    typedef std::list<Dsymbol*> DsymbolList;
    // dsymbols that need to be resolved
    DsymbolList resolveList;
    // dsymbols that need to be declared
    DsymbolList declareList;
    // dsymbols that need constant initializers constructed
    DsymbolList constInitList;
    // dsymbols that need definitions
    DsymbolList defineList;

    // static ctors/dtors/unittests
    typedef std::vector<FuncDeclaration*> FuncDeclVector;
    FuncDeclVector ctors;
    FuncDeclVector dtors;
    FuncDeclVector unitTests;

    // for inline asm
    IRAsmBlock* asmBlock;

    // dwarf dbg stuff
    // 'used' array solely for keeping a reference to globals
    std::vector<LLConstant*> usedArray;
    LLGlobalVariable* dwarfCUs;
    LLGlobalVariable* dwarfSPs;
    LLGlobalVariable* dwarfGVs;
};

template <typename InputIterator>
CallOrInvoke* IRState::CreateCallOrInvoke(LLValue* Callee, InputIterator ArgBegin, InputIterator ArgEnd, const char* Name)
{
    llvm::BasicBlock* pad;
    if(pad = func()->landingPad.get())
    {
        // intrinsics don't support invoking
        LLFunction* funcval = llvm::dyn_cast<LLFunction>(Callee);
        if (funcval && funcval->isIntrinsic())
        {
            return new CallOrInvoke_Call(ir->CreateCall(Callee, ArgBegin, ArgEnd, Name));
        }

        llvm::BasicBlock* postinvoke = llvm::BasicBlock::Create("postinvoke", topfunc(), scopeend());
        llvm::InvokeInst* invoke = ir->CreateInvoke(Callee, postinvoke, pad, ArgBegin, ArgEnd, Name);
        scope() = IRScope(postinvoke, scopeend());
        return new CallOrInvoke_Invoke(invoke);
    }
    else
        return new CallOrInvoke_Call(ir->CreateCall(Callee, ArgBegin, ArgEnd, Name));
}

#endif // LLVMDC_GEN_IRSTATE_H
