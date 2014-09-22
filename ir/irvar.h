//===-- ir/irdsymbol.h - Codegen state for D vars/fields/params -*- C++ -*-===//
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
};

// represents a global variable
struct IrGlobal : IrVar
{
    IrGlobal(VarDeclaration* v)
        : IrVar(v), type(0), constInit(0) { }
    IrGlobal(VarDeclaration* v, llvm::Type *type, llvm::Constant* constInit = 0)
        : IrVar(v), type(type), constInit(constInit) { }

    llvm::Type *type;
    llvm::Constant* constInit;
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
    IrField(VarDeclaration* v);
    void setAggrIndex(unsigned aggrIndex);

    unsigned index;
    unsigned unionOffset;

    llvm::Constant* getDefaultInit();

protected:
    /// FIXME: only used for StructLiteralsExps
    llvm::Constant* constInit;
};

IrVar *getIrVar(VarDeclaration *decl);
llvm::Value *getIrValue(VarDeclaration *decl);
bool isIrVarCreated(VarDeclaration *decl);

IrGlobal *getIrGlobal(VarDeclaration *decl, bool create = false);
bool isIrGlobalCreated(VarDeclaration *decl);

IrLocal *getIrLocal(VarDeclaration *decl, bool create = false);
bool isIrLocalCreated(VarDeclaration *decl);

IrParameter *getIrParameter(VarDeclaration *decl, bool create = false);
bool isIrParameterCreated(VarDeclaration *decl);

IrField *getIrField(VarDeclaration *decl, bool create = false);
bool isIrFieldCreated(VarDeclaration *decl);

#endif
