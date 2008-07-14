#ifndef LLVMDC_IR_IRFUNCTION_H
#define LLVMDC_IR_IRFUNCTION_H

#include "ir/ir.h"
#include "ir/irlandingpad.h"

#include <vector>
#include <stack>
#include <map>

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

    // pushes a unique label scope of the given name
    void pushUniqueLabelScope(const char* name);
    // pops a label scope
    void popLabelScope();

    // gets the string under which the label's BB
    // is stored in the labelToBB map.
    // essentially prefixes ident by the strings in labelScopes
    std::string getScopedLabelName(const char* ident);

    // label to basic block lookup
    typedef std::map<std::string, llvm::BasicBlock*> LabelToBBMap;
    LabelToBBMap labelToBB;

    // landing pads for try statements
    IRLandingPad landingPad;

    IrFunction(FuncDeclaration* fd);

private:
    // prefix for labels and gotos
    // used for allowing labels to be emitted twice
    std::vector<std::string> labelScopes;

    // next unique id stack
    std::stack<int> nextUnique;
};

#endif
