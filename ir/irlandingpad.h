//===-- ir/irlandingpad.h - Codegen state for EH blocks ---------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// State kept while doing codegen for a single "EH block" consisting of
// of several catch/finally/cleanup clauses. Handles nesting of these blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRLANDINGPADINFO_H
#define LDC_IR_IRLANDINGPADINFO_H

#include "statement.h"
#include "ir/ir.h"
#include <deque>
#include <stack>

namespace llvm {
    class Type;
    class Value;
    class BasicBlock;
    class Function;
}

// holds information about a single catch
struct IRLandingPadCatchInfo
{
    // default constructor for being able to store in a vector
    IRLandingPadCatchInfo() :
        target(NULL), end(0), catchStmt(NULL), catchType(0)
    {}

    IRLandingPadCatchInfo(Catch* catchStmt, llvm::BasicBlock* end);

    // codegen the catch block
    void toIR();

    llvm::BasicBlock *target;
    llvm::BasicBlock *end;
    Catch *catchStmt;
    ClassDeclaration *catchType;
};

// holds information about a single try-catch-inally block
struct IRLandingPadScope
{
    IRLandingPadScope() : target(0), finallyBody(0) {}

    // the target for invokes
    llvm::BasicBlock *target;
    // information about catch blocks
    std::deque<IRLandingPadCatchInfo> catches;
    // the body of finally
    Statement *finallyBody;
};


// holds information about all possible catch and finally actions
// and can emit landing pads to be called from the unwind runtime
struct IRLandingPad
{
    IRLandingPad() : catch_var(NULL) {}

    // creates a new landing pad according to given infos
    // and the ones on the stack. also stores it as invoke target
    void push(llvm::BasicBlock* inBB);

    // add catch information, will be used in next call to push
    void addCatch(Catch* catchstmt, llvm::BasicBlock* end);
    // add finally information, will be used in next call to push
    void addFinally(Statement* finallystmt);

    // builds the most recently constructed landing pad
    // and the catch blocks, then pops the landing pad bb
    // and its infos
    void pop();

    // gets the current landing pad
    llvm::BasicBlock* get();

    // creates or gets storage for exception object
    llvm::Value* getExceptionStorage();

private:
    // constructs the landing pad
    void constructLandingPad(IRLandingPadScope scope);

    // information about try-catch-finally blocks
    std::stack<IRLandingPadScope> scopeStack;
    IRLandingPadScope unpushedScope;

    // storage for the catch variable
    llvm::Value* catch_var;
};

#endif
