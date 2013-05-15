//===-- ir/ir.h - Base definitions for codegen metadata ---------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Declares the base class for all codegen info classes and the top-level Ir
// class.
//
//===----------------------------------------------------------------------===//


#ifndef LDC_IR_IR_H
#define LDC_IR_IR_H

#include "root.h"
#include "ir/irforw.h"
#include <deque>

struct IRState;
struct IrFunction;

class Ir
{
public:
    Ir();

    void setState(IRState* p)   { irs = p; }
    IRState* getState()         { return irs; }

    void addFunctionBody(IrFunction* f);
    void emitFunctionBodies();

private:
    IRState* irs;

    std::deque<IrFunction*> functionbodies;
};

#endif
