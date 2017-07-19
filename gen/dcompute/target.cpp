//===-- gen/dcompute/target.cpp -------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dcompute/target.h"
#include "ddmd/dsymbol.h"
#include "ddmd/module.h"
#include "ddmd/mars.h"
#include "ddmd/module.h"
#include "ddmd/scope.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/llvmhelpers.h"
#include "gen/runtime.h"
#include <string>

void DComputeTarget::doCodeGen(Module *m) {

  // process module members
  for (unsigned k = 0; k < m->members->dim; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym, _ir);
  }

  if (global.errors)
    fatal();
}

void DComputeTarget::emit(Module *m) {
  // Reset the global ABI to the target's ABI. Necessary because we have
  // multiple ABI we are trying to target. Also reset gIR. These are both
  // reused. Somewhat of a HACK.
  gABI = abi;
  gIR = _ir;
  doCodeGen(m);
}

void DComputeTarget::writeModule() {
  addMetadata();

  char filename[32];
  const char *fmt = "kernels_%s%d_%d.%s";
  int len = sprintf(filename, fmt, short_name, tversion,
                    global.params.is64bit ? 64 : 32, binSuffix);
  filename[len] = '\0';
  const char *fullname = FileName::combine(global.params.objdir, filename);

  setGTargetMachine();
  ::writeModule(&_ir->module, fullname);

  delete _ir;
  _ir = nullptr;
}
