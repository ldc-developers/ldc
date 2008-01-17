#ifndef LLVMDC_IR_IRFUNCTION_H
#define LLVMDC_IR_IRFUNCTION_H

#include "ir/ir.h"

#include <vector>

// represents a finally block
struct IrFinally
{
    llvm::BasicBlock* bb;
    llvm::BasicBlock* retbb;

    IrFinally();
    IrFinally(llvm::BasicBlock* b, llvm::BasicBlock* rb);
};

// represents a function
struct IrFunction : IrBase
{
    llvm::Function* func;
    llvm::Instruction* allocapoint;
    FuncDeclaration* decl;
    TypeFunction* type;

    // finally blocks
    typedef std::vector<IrFinally> FinallyVec;
    FinallyVec finallys;
    llvm::Value* finallyretval;

    bool queued;
    bool defined;
    llvm::Value* retArg;
    llvm::Value* thisVar;
    llvm::Value* nestedVar;
    llvm::Value* _arguments;
    llvm::Value* _argptr;
    llvm::Constant* dwarfSubProg;

    IrFunction(FuncDeclaration* fd);
    virtual ~IrFunction();
};

#endif
