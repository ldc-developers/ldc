#ifndef LLVMDC_GEN_IRSTATE_H
#define LLVMDC_GEN_IRSTATE_H

#include <vector>
#include <list>
#include <map>

#include "root.h"
#include "aggregate.h"

#include "ir/irfunction.h"
#include "ir/irstruct.h"
#include "ir/irvar.h"
#include "ir/irsymbol.h"
#include "ir/irtype.h"

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
struct TryFinallyStatement;

struct IrModule;

// represents a scope
struct IRScope
{
    llvm::BasicBlock* begin;
    llvm::BasicBlock* end;
    LLVMBuilder builder;

    IRScope();
    IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e);
};

// scope for loops
struct IRLoopScope : IRScope
{
    // generating statement
    Statement* s;
    // the try of a TryFinally that encloses the loop
    TryFinallyStatement* enclosingtryfinally;

    IRLoopScope();
    IRLoopScope(Statement* s, TryFinallyStatement* enclosingtryfinally, llvm::BasicBlock* b, llvm::BasicBlock* e);
};

struct IRBuilderHelper
{
    IRState* state;
    LLVMBuilder* operator->();
};

struct IRExp
{
    Expression* e1;
    Expression* e2;
    DValue* v;
    IRExp();
    IRExp(Expression* l, Expression* r, DValue* val);
};

// represents the module
struct IRState
{
    IRState();

    // module
    Module* dmodule;
    llvm::Module* module;

    // interface info type, used in DtoInterfaceInfoType
    llvm::StructType* interfaceInfoType;

    // ir data associated with DMD Dsymbol nodes 
    std::map<Dsymbol*, IrDsymbol> irDsymbol;

    // ir data associated with DMD Type instances
    std::map<Type*, IrType> irType;

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

    // expression l/r value handling
    typedef std::vector<IRExp> ExpVec;
    ExpVec exps;
    IRExp* topexp();

    // basic block scopes
    std::vector<IRScope> scopes;
    IRScope& scope();
    llvm::BasicBlock* scopebb();
    llvm::BasicBlock* scopeend();
    bool scopereturned();

    // loop blocks
    typedef std::vector<IRScope> BBVec;
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
};

#endif // LLVMDC_GEN_IRSTATE_H
