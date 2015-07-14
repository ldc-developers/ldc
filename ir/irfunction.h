//===-- ir/irfunction.h - Codegen state for D functions ---------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Represents the status of a D function/method/... on its way through the
// codegen process.
//
//===----------------------------------------------------------------------===//


#ifndef LDC_IR_IRFUNCTION_H
#define LDC_IR_IRFUNCTION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "gen/llvm.h"
#include "ir/irlandingpad.h"
#include "ir/irfuncty.h"
#include <map>
#include <stack>
#include <vector>

class Statement;
struct EnclosingTryFinally;
struct IRState;

// scope statements that can be target of jumps
// includes loops, switch, case, labels
struct IRTargetScope
{
    // generating statement
    Statement* s;

    // the try-finally block that encloses the statement
    EnclosingTryFinally* enclosinghandler;

    llvm::BasicBlock* breakTarget;
    llvm::BasicBlock* continueTarget;

    /// If true, the breakTarget is only considered if it is explicitly
    /// specified (via s), and not for unqualified "break;" statements.
    bool onlyLabeledBreak;

    IRTargetScope();
    IRTargetScope(
        Statement* s,
        EnclosingTryFinally* enclosinghandler,
        llvm::BasicBlock* continueTarget,
        llvm::BasicBlock* breakTarget,
        bool onlyLabeledBreak = false
    );
};

struct FuncGen
{
    FuncGen();

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

    // loop blocks
    typedef std::vector<IRTargetScope> TargetScopeVec;
    TargetScopeVec targetScopes;

    // landing pads for try statements
    IRLandingPad landingPadInfo;
    llvm::BasicBlock* landingPad;

    void pushToElemScope();
    void popToElemScope();

    void pushTemporaryToDestruct(VarDeclaration* vd);
    bool hasTemporariesToDestruct();
    void destructAllTemporaries();
    void destructAllTemporariesAndRestoreStack();
    // pushes a landing pad which needs to be popped after the
    // following invoke instruction
    void prepareToDestructAllTemporariesOnThrow(IRState* irState);

private:
    // prefix for labels and gotos
    // used for allowing labels to be emitted twice
    std::vector<std::string> labelScopes;

    // next unique id stack
    std::stack<int> nextUnique;

    int toElemScopeCounter;
    VarDeclarations temporariesToDestruct;
};

// represents a function
struct IrFunction
{
    // constructor
    IrFunction(FuncDeclaration* fd);

    // annotations
    void setNeverInline();
    void setAlwaysInline();

    llvm::Function* func;
    llvm::Instruction* allocapoint;
    FuncDeclaration* decl;
    TypeFunction* type;

    FuncGen* gen;

    bool queued;
    bool defined;

    llvm::Value* retArg; // return in ptr arg
    llvm::Value* thisArg; // class/struct 'this' arg
    llvm::Value* nestArg; // nested function 'this' arg

    llvm::Value* nestedVar; // alloca for the nested context of this function
    llvm::StructType* frameType; // type of nested context
    // number of enclosing functions with variables accessed by nested functions
    // (-1 if neither this function nor any enclosing ones access variables from enclosing functions)
    int depth;
    bool nestedContextCreated; // holds whether nested context is created

    llvm::Value* _arguments;
    llvm::Value* _argptr;

#if LDC_LLVM_VER >= 307
    llvm::DISubprogram* diSubprogram = nullptr;
    std::stack<llvm::DILexicalBlock*> diLexicalBlocks;
    typedef llvm::DenseMap<VarDeclaration*, llvm::DILocalVariable*> VariableMap;
#else
    llvm::DISubprogram diSubprogram;
    std::stack<llvm::DILexicalBlock> diLexicalBlocks;
    typedef llvm::DenseMap<VarDeclaration*, llvm::DIVariable> VariableMap;
#endif
    // Debug info for all variables
    VariableMap variableMap;

    IrFuncTy irFty;
};

IrFunction *getIrFunc(FuncDeclaration *decl, bool create = false);
bool isIrFuncCreated(FuncDeclaration *decl);

#endif
