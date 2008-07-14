#ifndef LLVMDC_IR_IRFUNCTION_H
#define LLVMDC_IR_IRFUNCTION_H

#include "ir/ir.h"
#include "ir/irlandingpad.h"

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
    llvm::AllocaInst* msgArg;

    // label to basic block lookup
    typedef std::map<std::string, llvm::BasicBlock*> LabelToBBMap;
    LabelToBBMap labelToBB;

    // landing pads for try statements
    IRLandingPad landingPad;

    IrFunction(FuncDeclaration* fd);
};

#endif
