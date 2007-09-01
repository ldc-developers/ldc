#ifndef LLVMDC_GEN_IRSTATE_H
#define LLVMDC_GEN_IRSTATE_H

#include <stack>
#include <vector>
#include <deque>

#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/Target/TargetData.h"

#include "root.h"

// global ir state for current module
struct IRState;
extern IRState* gIR;
extern llvm::TargetData* gTargetData;

struct TypeFunction;
struct TypeStruct;
struct ClassDeclaration;
struct FuncDeclaration;
struct Module;
struct TypeStruct;

// represents a scope
struct IRScope
{
    llvm::BasicBlock* begin;
    llvm::BasicBlock* end;
    bool returned;

    IRScope();
    IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e);
};

// represents a struct
struct IRStruct : Object
{
    typedef std::vector<const llvm::Type*> TypeVector;
    typedef std::vector<llvm::Constant*> ConstantVector;
    typedef std::vector<llvm::PATypeHolder> PATypeHolderVector;

public:
    IRStruct();
    IRStruct(TypeStruct*);
    virtual ~IRStruct();

    TypeStruct* type;
    TypeVector fields;
    ConstantVector inits;
    llvm::PATypeHolder recty;
};

// represents a clas
struct IRClass : Object
{
    // TODO
};

// represents the module
struct IRState : Object
{
    IRState();

    // module
    Module* dmodule;
    llvm::Module* module;

    // functions
    std::stack<llvm::Function*> funcs;
    llvm::Function* topfunc();
    std::stack<TypeFunction*> functypes;
    TypeFunction* topfunctype();
    llvm::Instruction* topallocapoint();

    // structs
    typedef std::vector<IRStruct> StructVector;
    StructVector structs;
    IRStruct& topstruct();

    // classes TODO move into IRClass
    typedef std::vector<ClassDeclaration*> ClassDeclVec;
    ClassDeclVec classes;
    typedef std::vector<FuncDeclaration*> FuncDeclVec;
    typedef std::vector<FuncDeclVec> ClassMethodVec;
    ClassMethodVec classmethods;
    typedef std::vector<bool> BoolVec;
    BoolVec queueClassMethods;

    // D main function
    bool emitMain;
    llvm::Function* mainFunc;

    // L-values
    bool inLvalue;
    typedef std::vector<llvm::Value*> LvalVec;
    LvalVec lvals;
    llvm::Value* toplval();

    // basic block scopes
    std::vector<IRScope> scopes;
    IRScope& scope();
    llvm::BasicBlock* scopebegin();
    llvm::BasicBlock* scopeend();
    llvm::BasicBlock* scopebb();
    bool scopereturned();

    // loop blocks
    typedef std::vector<IRScope> BBVec;
    BBVec loopbbs;

    // this holds the array being indexed or sliced so $ will work
    // might be a better way but it works. problem is I only get a
    // VarDeclaration for __dollar, but I can't see how to get the
    // array pointer from this :(
    LvalVec arrays;
};

#endif // LLVMDC_GEN_IRSTATE_H
