//===-- ir/irvar.h - Codegen state for D vars/fields/params -*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Classes for representing the status of D variables on their way though the
// codegen process.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRVAR_H
#define LDC_IR_IRVAR_H

#if LDC_LLVM_VER >= 303
#include "llvm/IR/Type.h"
#else
#include "llvm/Type.h"
#endif

#if LDC_LLVM_VER >= 305
#include "llvm/IR/DebugInfo.h"
#elif LDC_LLVM_VER >= 302
#include "llvm/DebugInfo.h"
#else
#include "llvm/Analysis/DebugInfo.h"
#endif

struct IrFuncTyArg;
class VarDeclaration;

struct IrVar
{
    IrVar(VarDeclaration* var)
        : V(var), value(0) { }
    IrVar(VarDeclaration* var, llvm::Value* value)
        : V(var), value(value) { }

    VarDeclaration* V;
    llvm::Value* value;

    // debug description
    llvm::DIVariable debugVariable;
    llvm::DISubprogram debugFunc;
};

// represents a global variable
struct IrGlobal : IrVar
{
    IrGlobal(VarDeclaration* v)
        : IrVar(v), type(0), constInit(0), nakedUse(false) { }
    IrGlobal(VarDeclaration* v, llvm::Type *type, llvm::Constant* constInit = 0)
        : IrVar(v), type(type), constInit(constInit), nakedUse(false) { }

    llvm::Type *type;
    llvm::Constant* constInit;

    // This var is used by a naked function.
    bool nakedUse;
};

// represents a local variable variable
struct IrLocal : IrVar
{
    IrLocal(VarDeclaration* v)
        : IrVar(v), nestedDepth(0), nestedIndex(-1) { }
    IrLocal(VarDeclaration* v, llvm::Value* value)
        : IrVar(v, value), nestedDepth(0), nestedIndex(-1) { }
    IrLocal(VarDeclaration* v, int nestedDepth, int nestedIndex)
        : IrVar(v), nestedDepth(nestedDepth), nestedIndex(nestedIndex) { }

    // Used for hybrid nested context creation.
    int nestedDepth;
    int nestedIndex;
};

// represents a function parameter
struct IrParameter : IrLocal
{
    IrParameter(VarDeclaration* v)
        : IrLocal(v), arg(0), isVthis(false) { }
    IrParameter(VarDeclaration* v, llvm::Value* value)
        : IrLocal(v, value), arg(0), isVthis(false) { }
    IrParameter(VarDeclaration* v, llvm::Value* value, IrFuncTyArg *arg, bool isVthis = false)
        : IrLocal(v, value), arg(arg), isVthis(isVthis) { }

    IrFuncTyArg *arg;
    bool isVthis; // true, if it is the 'this' parameter
};

// represents an aggregate field variable
struct IrField : IrVar
{
    IrField(VarDeclaration* v) : IrVar(v) {};
};

llvm::Value *getIrValue(VarDeclaration *decl);

#endif
