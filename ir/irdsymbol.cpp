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
#include "ir/irdsymbol.h"
#include "ir/irvar.h"

std::vector<IrDsymbol*> IrDsymbol::list;

void IrDsymbol::resetAll()
{
    Logger::println("resetting %zu Dsymbols", list.size());

    for (std::vector<IrDsymbol*>::iterator it = list.begin(), end = list.end(); it != end; ++it)
        (*it)->reset();
}

IrDsymbol::IrDsymbol()
{
    list.push_back(this);
    reset();
}

IrDsymbol::IrDsymbol(const IrDsymbol& s)
{
    list.push_back(this);
    irModule = s.irModule;
    irAggr   = s.irAggr;
    irFunc   = s.irFunc;
    irGlobal = s.irGlobal;
    irLocal  = s.irLocal;
    irField  = s.irField;
    resolved = s.resolved;
    declared = s.declared;
    initialized = s.initialized;
    defined  = s.defined;
}

IrDsymbol::~IrDsymbol()
{
    if (this == list.back())
    {
        list.pop_back();
        return;
    }

    std::vector<IrDsymbol*>::iterator it = std::find(list.rbegin(), list.rend(), this).base();
    // base() returns the iterator _after_ the found position
    list.erase(--it);
}

void IrDsymbol::reset()
{
    irModule = NULL;
    irAggr   = NULL;
    irFunc   = NULL;
    irGlobal = NULL;
    irLocal  = NULL;
    irField  = NULL;
    resolved = declared = initialized = defined = false;
}

bool IrDsymbol::isSet()
{
    return irAggr || irFunc || irGlobal || irLocal || irField;
}

IrVar* IrDsymbol::getIrVar()
{
    assert(irGlobal || irLocal || irField);
    return irGlobal ? static_cast<IrVar*>(irGlobal) : irLocal ? static_cast<IrVar*>(irLocal) : static_cast<IrVar*>(irField);
}

llvm::Value*& IrDsymbol::getIrValue() { return getIrVar()->value; }
