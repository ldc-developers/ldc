//===-- targetCUDA.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/target.h"
#include "dsymbol.h"
#include "module.h"
#include "gen/llvmhelpers.h"
#include "id.h"
#include "mars.h"
#include "module.h"
#include "scope.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include <string>
DComputeTarget::DComputeTarget(llvm::LLVMContext &c, int v)
    : ctx(c), tversion(v) {}

void DComputeTarget::doCodeGen(Module *m) {

  // process module members
  for (unsigned k = 0; k < m->members->dim; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym, _ir, this);
  }

  if (global.errors) {
    fatal();
  }
}

void DComputeTarget::emit(Module *m) {
  gABI = abi;
  gIR = _ir;
  doCodeGen(m);
}

void DComputeTarget::writeModule() {
  addMetadata();
  // TODO: make a command line switch for the and do it properly so as to not
  // cross @conpute code with non-@compute code
  // insertBitcodeFiles(_ir->module, _ir->context(),
  //                   *global.params.bitcodeFiles);
  const char *filename;

  char tmp[20];
  const char *fmt = "kernels_%s%d_%d";
  int len = sprintf(tmp, fmt, (target == 1) ? "ocl" : "cuda", tversion,
                    global.params.is64bit ? 64 : 32);
  tmp[len] = '\0';
  filename = FileName::forceExt(tmp, binSuffix);
  setGTargetMachine();
  ::writeModule(&_ir->module, filename);

  global.params.objfiles->push(filename);
  delete _ir;
  _ir = nullptr;
}
