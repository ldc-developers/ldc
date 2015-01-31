//===-- irmetadata.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irmetadata.h"

#include "aggregate.h"
#include "declaration.h"
#include "module.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "ir/iraggr.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "ir/irvar.h"
#include <unordered_map>

std::unordered_map<Dsymbol*, IrMetadata> metadata;

static
IrMetadata* get(Dsymbol* sym)
{
    return &metadata[sym];
}

// default if not found
static
IrMetadata find(Dsymbol* sym)
{
    auto it = metadata.find(sym);
    if (it == metadata.end())
        return IrMetadata();
    return it->second;
}

IrMetadata::IrMetadata()
{
    irData  = NULL;
    m_type  = NotSet;
    m_state = Initial;
}

void IrMetadata::resetAll()
{
    Logger::println("resetting %zu Dsymbols", metadata.size());
    metadata.clear();
}

void IrMetadata::setResolved()
{
    if (m_state < Resolved)
        m_state = Resolved;
}

void IrMetadata::setDeclared()
{
    if (m_state < Declared)
        m_state = Declared;
}

void IrMetadata::setInitialized()
{
    if (m_state < Initialized)
        m_state = Initialized;
}

void IrMetadata::setDefined()
{
    if (m_state < Defined)
        m_state = Defined;
}

//////////////////////////////////////////////////////////////////////////////

IrMetadata *getIrMetadata(Dsymbol *sym)
{
    return get(sym);
}

//////////////////////////////////////////////////////////////////////////////

IrVar *getIrVar(VarDeclaration *decl)
{
    IrMetadata *irm = get(decl);
    int t = irm->type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::GlobalType ||
           t == IrMetadata::LocalType ||
           t == IrMetadata::ParamterType ||
           t == IrMetadata::FieldType);
    assert(irm->irVar != NULL);
    return irm->irVar;
}

bool isIrVarCreated(VarDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::GlobalType ||
           t == IrMetadata::LocalType ||
           t == IrMetadata::ParamterType ||
           t == IrMetadata::FieldType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrGlobal *getIrGlobal(VarDeclaration *decl, bool create)
{
    IrMetadata *irm = get(decl);
    if (irm->type() == IrMetadata::NotSet && create)
    {
        assert(irm->irGlobal == NULL);
        irm->irGlobal = new IrGlobal(decl);
        irm->m_type = IrMetadata::GlobalType;
    }
    assert(irm->type() == IrMetadata::GlobalType);
    assert(irm->irGlobal != NULL);
    return irm->irGlobal;
}

bool isIrGlobalCreated(VarDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::GlobalType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrLocal *getIrLocal(VarDeclaration *decl, bool create)
{
    IrMetadata *irm = get(decl);
    if (irm->type() == IrMetadata::NotSet && create)
    {
        assert(irm->irLocal == NULL);
        irm->irLocal = new IrLocal(decl);
        irm->m_type = IrMetadata::LocalType;
    }
    assert(irm->type() == IrMetadata::LocalType ||
           irm->type() == IrMetadata::ParamterType);
    assert(irm->irLocal != NULL);
    return irm->irLocal;
}

bool isIrLocalCreated(VarDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::LocalType ||
           t == IrMetadata::ParamterType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrParameter *getIrParameter(VarDeclaration *decl, bool create)
{
    IrMetadata *irm = get(decl);
    if (irm->type() == IrMetadata::NotSet && create)
    {
        assert(irm->irParam == NULL);
        irm->irParam = new IrParameter(decl);
        irm->m_type = IrMetadata::ParamterType;
    }
    assert(irm->type() == IrMetadata::ParamterType);
    assert(irm->irParam != NULL);
    return irm->irParam;
}


bool isIrParameterCreated(VarDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::ParamterType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrField *getIrField(VarDeclaration *decl, bool create)
{
    IrMetadata *irm = get(decl);
    if (irm->type() == IrMetadata::NotSet && create)
    {
        assert(irm->irField == NULL);
        irm->irField = new IrField(decl);
        irm->m_type = IrMetadata::FieldType;
    }
    assert(irm->type() == IrMetadata::FieldType);
    assert(irm->irField != NULL);
    return irm->irField;
}

bool isIrFieldCreated(VarDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::FieldType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrFunction *getIrFunc(FuncDeclaration *decl, bool create)
{
    IrMetadata *irm = get(decl);
    if (irm->type() == IrMetadata::NotSet && create)
    {
        assert(irm->irFunc == NULL);
        irm->irFunc = new IrFunction(decl);
        irm->m_type = IrMetadata::FuncType;
    }
    assert(irm->type() == IrMetadata::FuncType);
    assert(irm->irFunc != NULL);
    return irm->irFunc;
}

bool isIrFuncCreated(FuncDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::FuncType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create)
{
    IrMetadata *irm = get(decl);
    if (irm->type() == IrMetadata::NotSet && create)
    {
        assert(irm->irAggr == NULL);
        irm->irAggr = new IrAggr(decl);
        irm->m_type = IrMetadata::AggrType;
    }
    assert(irm->type() == IrMetadata::AggrType);
    assert(irm->irAggr != NULL);
    return irm->irAggr;
}

bool isIrAggrCreated(AggregateDeclaration *decl)
{
    int t = find(decl).type();
    assert(t == IrMetadata::NotSet ||
           t == IrMetadata::AggrType);
    return t != IrMetadata::NotSet;
}

//////////////////////////////////////////////////////////////////////////////

IrModule *getIrModule(Module *m)
{
    if (m == NULL)
        m = gIR->func()->decl->getModule();
    assert(m && "null module");
    IrMetadata *irm = get(m);
    if (irm->type() == IrMetadata::NotSet)
    {
        irm->irModule = new IrModule(m, m->srcfile->toChars());
        irm->m_type = IrMetadata::ModuleType;
    }
    assert(irm->type() == IrMetadata::ModuleType);
    assert(irm->irModule != NULL);
    return irm->irModule;
}
