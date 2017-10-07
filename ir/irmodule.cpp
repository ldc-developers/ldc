//===-- irmodule.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irmodule.h"
#include "module.h"
#include "gen/llvm.h"
#include "gen/irstate.h"
#include "gen/mangling.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"

IrModule::IrModule(Module *module) : M(module) {}

llvm::GlobalVariable *IrModule::moduleInfoSymbol() {
  if (moduleInfoVar) {
    return moduleInfoVar;
  }

  const auto irMangle = getIRMangledModuleInfoSymbolName(M);

  moduleInfoVar = new llvm::GlobalVariable(
      gIR->module, llvm::StructType::create(gIR->context()), false,
      llvm::GlobalValue::ExternalLinkage, nullptr, irMangle);
  return moduleInfoVar;
}

IrModule *getIrModule(Module *m) {
  if (!m) {
    m = gIR->func()->decl->getModule();
  }

  assert(m && "null module");
  if (m->ir->m_type == IrDsymbol::NotSet) {
    m->ir->irModule = new IrModule(m);
    m->ir->m_type = IrDsymbol::ModuleType;
  }

  assert(m->ir->m_type == IrDsymbol::ModuleType);
  return m->ir->irModule;
}
