#ifndef LLVMDC_GEN_IRSTATE_H
#define LLVMDC_GEN_IRSTATE_H

#include <stack>
#include <vector>
#include <deque>
#include <map>

#include "root.h"

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

/*
struct LLVMValue
{
    std::vector<llvm::Value*> vals;
};
*/

// represents a scope
struct IRScope
{
    llvm::BasicBlock* begin;
    llvm::BasicBlock* end;
    LLVMBuilder builder;

    IRScope();
    IRScope(llvm::BasicBlock* b, llvm::BasicBlock* e);
};

// represents a struct or class
struct IRStruct
{
    struct Offset
    {
        VarDeclaration* var;
        llvm::Constant* init;

        Offset(VarDeclaration* v, llvm::Constant* i)
        : var(v), init(i) {}
    };

    typedef std::vector<FuncDeclaration*> FuncDeclVector;
    typedef std::multimap<unsigned, Offset> OffsetMap;

public:
    IRStruct();
    IRStruct(Type*);

    Type* type;
    llvm::PATypeHolder recty;
    FuncDeclVector funcs;
    bool queueFuncs;

    OffsetMap offsets;
};

// represents a finally block
struct IRFinally
{
    llvm::BasicBlock* bb;
    llvm::BasicBlock* retbb;

    IRFinally();
    IRFinally(llvm::BasicBlock* b, llvm::BasicBlock* rb);
};

// represents a function
struct IRFunction
{
    llvm::Function* func;
    llvm::Instruction* allocapoint;
    FuncDeclaration* decl;
    TypeFunction* type;

    // finally blocks
    typedef std::vector<IRFinally> FinallyVec;
    FinallyVec finallys;
    llvm::Value* finallyretval;

    IRFunction(FuncDeclaration*);
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
    llvm::Value* v;
    IRExp();
    IRExp(Expression* l, Expression* r, llvm::Value* val);
};

// represents the module
struct IRState
{
    IRState();

    // module
    Module* dmodule;
    llvm::Module* module;

    // functions
    typedef std::vector<IRFunction> FunctionVector;
    FunctionVector functions;
    IRFunction& func();

    llvm::Function* topfunc();
    TypeFunction* topfunctype();
    llvm::Instruction* topallocapoint();

    // structs
    typedef std::vector<IRStruct> StructVector;
    StructVector structs;
    IRStruct& topstruct();

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
    std::vector<llvm::Value*> arrays;

    // builder helper
    IRBuilderHelper ir;

    // Dwarf debugging info
    llvm::GlobalVariable* dwarfCompileUnit;
};

#endif // LLVMDC_GEN_IRSTATE_H
