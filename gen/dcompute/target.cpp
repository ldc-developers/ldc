//===-- gen/dcompute/target.cpp -------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ddmd/dsymbol.h"
#include "ddmd/mars.h"
#include "ddmd/module.h"
#include "ddmd/scope.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/dcompute/target.h"
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

  std::string filename;
  llvm::raw_string_ostream os(filename);
  os << "kernels_" << short_name << tversion << '_'
     << (global.params.is64bit ? 64 : 32) << '.' << binSuffix;

  const char *path = FileName::combine(global.params.objdir, os.str().c_str());

  setGTargetMachine();
  ::writeModule(&_ir->module, path);

  delete _ir;
  _ir = nullptr;
}
