//===-- irvar.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "declaration.h"
#include "gen/irstate.h"
#include "ir/irvar.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrField::IrField(VarDeclaration* v) : IrVar(v)
{
    assert(V->ir.irField == NULL && "field for this variable already exists");
    V->ir.irField = this;

    if (v->aggrIndex)
    {
        index = v->aggrIndex;
        unionOffset = 0;
    }
    else
    {
        index = 0;
        unionOffset = v->offset;
    }
}

extern LLConstant* get_default_initializer(VarDeclaration* vd, Initializer* init);

llvm::Constant* IrField::getDefaultInit()
{
    if (constInit)
        return constInit;
    constInit = get_default_initializer(V, V->init);
    return constInit;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
