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
#include "ir/irdsymbol.h"
#include "ir/irvar.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrVar *getIrVar(VarDeclaration *decl)
{
    assert(isIrVarCreated(decl));
    assert(decl->ir.irVar != NULL);
    return decl->ir.irVar;
}

llvm::Value *getIrValue(VarDeclaration *decl)
{
    return getIrVar(decl)->value;
}

bool isIrVarCreated(VarDeclaration *decl)
{
    int t = decl->ir.type();
    bool isIrVar = t == IrDsymbol::GlobalType ||
                   t == IrDsymbol::LocalType ||
                   t == IrDsymbol::ParamterType ||
                   t == IrDsymbol::FieldType;
    assert(isIrVar || t == IrDsymbol::NotSet);
    return isIrVar;
}

//////////////////////////////////////////////////////////////////////////////

IrGlobal *getIrGlobal(VarDeclaration *decl, bool create)
{
    if (!isIrGlobalCreated(decl) && create)
    {
        assert(decl->ir.irGlobal == NULL);
        decl->ir.irGlobal = new IrGlobal(decl);
        decl->ir.m_type = IrDsymbol::GlobalType;
    }
    assert(decl->ir.irGlobal != NULL);
    return decl->ir.irGlobal;
}

bool isIrGlobalCreated(VarDeclaration *decl)
{
    int t = decl->ir.type();
    assert(t == IrDsymbol::GlobalType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::GlobalType;
}

//////////////////////////////////////////////////////////////////////////////

IrLocal *getIrLocal(VarDeclaration *decl, bool create)
{
    if (!isIrLocalCreated(decl) && create)
    {
        assert(decl->ir.irLocal == NULL);
        decl->ir.irLocal = new IrLocal(decl);
        decl->ir.m_type = IrDsymbol::LocalType;
    }
    assert(decl->ir.irLocal != NULL);
    return decl->ir.irLocal;
}

bool isIrLocalCreated(VarDeclaration *decl)
{
    int t = decl->ir.type();
    assert(t == IrDsymbol::LocalType || t == IrDsymbol::ParamterType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::LocalType || t == IrDsymbol::ParamterType;
}

//////////////////////////////////////////////////////////////////////////////

IrParameter *getIrParameter(VarDeclaration *decl, bool create)
{
    if (!isIrParameterCreated(decl) && create)
    {
        assert(decl->ir.irParam == NULL);
        decl->ir.irParam = new IrParameter(decl);
        decl->ir.m_type = IrDsymbol::ParamterType;
    }
    return decl->ir.irParam;
}

bool isIrParameterCreated(VarDeclaration *decl)
{
    int t = decl->ir.type();
    assert(t == IrDsymbol::ParamterType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::ParamterType;
}

//////////////////////////////////////////////////////////////////////////////

IrField *getIrField(VarDeclaration *decl, bool create)
{
    if (!isIrFieldCreated(decl) && create)
    {
        assert(decl->ir.irField == NULL);
        decl->ir.irField = new IrField(decl);
        decl->ir.m_type = IrDsymbol::FieldType;
    }
    assert(decl->ir.irField != NULL);
    return decl->ir.irField;
}

bool isIrFieldCreated(VarDeclaration *decl)
{
    int t = decl->ir.type();
    assert(t == IrDsymbol::FieldType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::FieldType;
}
