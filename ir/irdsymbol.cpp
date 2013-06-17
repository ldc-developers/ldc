//===-- irdsymbol.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "gen/logger.h"
#include "ir/ir.h"
#include "ir/irdsymbol.h"
#include "ir/irvar.h"

std::set<IrDsymbol*> IrDsymbol::list;

void IrDsymbol::resetAll()
{
    Logger::println("resetting %zu Dsymbols", list.size());
    std::set<IrDsymbol*>::iterator it;
    for(it = list.begin(); it != list.end(); ++it)
        (*it)->reset();
}

IrDsymbol::IrDsymbol()
{
    bool incr = list.insert(this).second;
    assert(incr);
    reset();
}

IrDsymbol::IrDsymbol(const IrDsymbol& s)
{
    bool incr = list.insert(this).second;
    assert(incr);
    DModule = s.DModule;
    irModule = s.irModule;
    irAggr = s.irAggr;
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
    irAggr = NULL;
    irFunc = NULL;
    resolved = declared = initialized = defined = false;
    irGlobal = NULL;
    irLocal = NULL;
    irField = NULL;
}

bool IrDsymbol::isSet()
{
    return (irAggr || irFunc || irGlobal || irLocal || irField);
}

IrVar* IrDsymbol::getIrVar()
{
    assert(irGlobal || irLocal || irField);
    return irGlobal ? static_cast<IrVar*>(irGlobal) : irLocal ? static_cast<IrVar*>(irLocal) : static_cast<IrVar*>(irField);
}

llvm::Value*& IrDsymbol::getIrValue() { return getIrVar()->value; }
