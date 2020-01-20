//===-- irmodule.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irmodule.h"

#include "dmd/module.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
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

  moduleInfoVar =
      declareGlobal(Loc(), gIR->module,
                    llvm::StructType::create(gIR->context()), irMangle, false);

  // Like DMD, declare as weak - don't pull in the object file just because of
  // the import, i.e., use null if the object isn't pulled in by something else.
  // FIXME: Disabled for MSVC targets, because LLD (9.0.1) fails with duplicate
  //        symbol errors - MS linker works (tested for Win64).
  if (!global.params.targetTriple->isWindowsMSVCEnvironment()) {
    moduleInfoVar->setLinkage(LLGlobalValue::ExternalWeakLinkage);
  }

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
