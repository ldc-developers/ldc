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
    Logger::println("resetting %llu Dsymbols", static_cast<unsigned long long>(list.size()));

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
    irData  = s.irData;
    m_type  = s.m_type;
    m_state = s.m_state;
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
    irData  = NULL;
    m_type  = NotSet;
    m_state = Initial;
}

void IrDsymbol::setResolved()
{
    if (m_state < Resolved)
        m_state = Resolved;
}

void IrDsymbol::setDeclared()
{
    if (m_state < Declared)
        m_state = Declared;
}

void IrDsymbol::setInitialized()
{
    if (m_state < Initialized)
        m_state = Initialized;
}

void IrDsymbol::setDefined()
{
    if (m_state < Defined)
        m_state = Defined;
}
