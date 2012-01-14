#ifndef LDC_IR_IRVAR_H
#define LDC_IR_IRVAR_H

#include "ir/ir.h"
#include "llvm/Type.h"

struct IrFuncTyArg;

struct IrVar : IrBase
{
    IrVar(VarDeclaration* var);

    VarDeclaration* V;
    llvm::Value* value;
};

// represents a global variable
struct IrGlobal : IrVar
{
    IrGlobal(VarDeclaration* v);

    llvm::Type *type;
    llvm::Constant* constInit;
};

// represents a local variable variable
struct IrLocal : IrVar
{
    IrLocal(VarDeclaration* v);

    bool byref;         // Not used for -nested-ctx=array
    int nestedDepth;    // ditto
    int nestedIndex;
};

// represents a function parameter
struct IrParameter : IrLocal
{
    IrParameter(VarDeclaration* v);
    IrFuncTyArg *arg;
};

// represents an aggregate field variable
struct IrField : IrVar
{
    IrField(VarDeclaration* v);

    unsigned index;
    unsigned unionOffset;

    llvm::Constant* getDefaultInit();

protected:
    /// FIXME: only used for StructLiteralsExps
    llvm::Constant* constInit;
};

#endif
