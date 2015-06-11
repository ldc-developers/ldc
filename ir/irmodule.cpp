//===-- irmodule.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "module.h"
#include "gen/llvm.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irmodule.h"

IrModule::IrModule(Module* module, const char* srcfilename)
{
    M = module;
}

IrModule::~IrModule()
{
}

IrModule *getIrModule(Module *m)
{
    if (m == NULL)
        m = gIR->func()->decl->getModule();
    assert(m && "null module");
    if (m->ir.m_type == IrDsymbol::NotSet)
    {
        m->ir.irModule = new IrModule(m, m->srcfile->toChars());
        m->ir.m_type = IrDsymbol::ModuleType;
    }
    assert(m->ir.m_type == IrDsymbol::ModuleType);
    return m->ir.irModule;
}
