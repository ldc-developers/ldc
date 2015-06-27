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

IrModule::IrModule(Module *module, const char *srcfilename)
    : M(module), moduleInfoVar_(0) {}

IrModule::~IrModule() {}

llvm::GlobalVariable *IrModule::moduleInfoSymbol() {
    if (moduleInfoVar_) return moduleInfoVar_;

    std::string name("_D");
    name.append(mangle(M));
    name.append("12__ModuleInfoZ");

    moduleInfoVar_ = new llvm::GlobalVariable(
        gIR->module, llvm::StructType::create(gIR->context()), false,
        llvm::GlobalValue::ExternalLinkage, NULL, name);
    return moduleInfoVar_;
}

IrModule *getIrModule(Module *m) {
    if (!m) m = gIR->func()->decl->getModule();

    assert(m && "null module");
    if (m->ir.m_type == IrDsymbol::NotSet) {
        m->ir.irModule = new IrModule(m, m->srcfile->toChars());
        m->ir.m_type = IrDsymbol::ModuleType;
    }

    assert(m->ir.m_type == IrDsymbol::ModuleType);
    return m->ir.irModule;
}
