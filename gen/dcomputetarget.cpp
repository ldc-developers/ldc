//===-- targetCUDA.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ddmd/dsymbol.h"
#include "ddmd/module.h"
#include "ddmd/mars.h"
#include "ddmd/module.h"
#include "ddmd/scope.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/dcomputetarget.h"
#include "gen/llvmhelpers.h"
#include "id.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include <string>

DComputeTarget::DComputeTarget(llvm::LLVMContext &c, int v, ID id,
                               const char *_short_name, const char *suffix,
                               TargetABI *a, std::array<int, 5> map)
    : ctx(c), tversion(v), target(id), short_name(_short_name),
      binSuffix(suffix), abi(a), mapping(map) {}

void DComputeTarget::doCodeGen(Module *m) {

  // process module members
  for (unsigned k = 0; k < m->members->dim; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym, _ir);
  }

  if (global.errors) {
    fatal();
  }
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

  char tmp[32];
  const char *fmt = "kernels_%s%d_%d.%s";
  int len = sprintf(tmp, fmt, short_name, tversion,
                    global.params.is64bit ? 64 : 32, binSuffix);
  tmp[len] = '\0';
  setGTargetMachine();
  ::writeModule(&_ir->module, tmp);


  delete _ir;
  _ir = nullptr;
}
