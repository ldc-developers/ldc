#ifndef LLVMDC_IR_IRFUNCTION_H
#define LLVMDC_IR_IRFUNCTION_H

#include "ir/ir.h"

#include <vector>

// represents a function
struct IrFunction : IrBase
{
    llvm::Function* func;
    llvm::Instruction* allocapoint;
    FuncDeclaration* decl;
    TypeFunction* type;

    bool queued;
    bool defined;
    llvm::Value* retArg;
    llvm::Value* thisVar;
    llvm::Value* nestedVar;
    llvm::Value* _arguments;
    llvm::Value* _argptr;
    llvm::Constant* dwarfSubProg;

    llvm::AllocaInst* srcfileArg;

    bool inVolatile;

    IrFunction(FuncDeclaration* fd);
    virtual ~IrFunction();
};

#endif
