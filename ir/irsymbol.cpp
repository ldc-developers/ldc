#include "gen/llvm.h"
#include "ir/ir.h"
#include "ir/irsymbol.h"
#include "ir/irvar.h"

std::set<IrDsymbol*> IrDsymbol::list;

void IrDsymbol::resetAll()
{
    std::set<IrDsymbol*>::iterator it;
    for(it = list.begin(); it != list.end(); ++it)
        (*it)->reset();
}

IrDsymbol::IrDsymbol()
{
    list.insert(this);
    reset();
}

IrDsymbol::IrDsymbol(const IrDsymbol& s)
{
    list.insert(this);
    DModule = s.DModule;
    irModule = s.irModule;
    irStruct = s.irStruct;
    irFunc = s.irFunc;
    resolved = s.resolved;
    declared = s.declared;
    initialized = s.initialized;
    defined = s.defined;
    irGlobal = s.irGlobal;
    irLocal = s.irLocal;
    irField = s.irField;
}

IrDsymbol::~IrDsymbol()
{
    list.erase(this);
}

void IrDsymbol::reset()
{
    DModule = NULL;
    irModule = NULL;
    irStruct = NULL;
    irFunc = NULL;
    resolved = declared = initialized = defined = false;
    irGlobal = NULL;
    irLocal = NULL;
    irField = NULL;
}

IrVar* IrDsymbol::getIrVar()
{
    assert(irGlobal || irLocal || irField);
    return irGlobal ? (IrVar*)irGlobal : irLocal ? (IrVar*)irLocal : (IrVar*)irField;
}

llvm::Value*& IrDsymbol::getIrValue() { return getIrVar()->value; }
