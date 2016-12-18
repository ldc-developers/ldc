//===-- dcompute/codegenmanager.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/dcomputecodegenmanager.h"
#include "ir/irdsymbol.h"
#include "driver/cl_options.h"
#include "gen/cl_helpers.h"
#include "llvm/Support/CommandLine.h"
#include "ddmd/errors.h"
#include <string>
#include <algorithm>
#include "driver/cl_options.h"

DComputeTarget *
DComputeCodeGenManager::createComputeTarget(const std::string &s) {
  int v;
  if (s.substr(0, 4) == "ocl-") {
    v = atoi(s.c_str() + 4);
    llvm::SmallVector<int, 6> versions = {100, 110, 120, 200, 210, 220};
    if (find(versions.begin(), versions.end(), v) != versions.end()) {
      return createOCLTarget(ctx, v);
    }
  } else if (s.substr(0, 5) == "cuda-") {
    v = atoi(s.c_str() + 5);
    llvm::SmallVector<int, 14> versions = {100, 110, 120, 130, 200, 210, 300,
                                           350, 370, 500, 520, 600, 610, 620};
    if (find(versions.begin(), versions.end(), v) != versions.end()) {
      return createOCLTarget(ctx, v);
    }
  }
  error(Loc(),
        "unrecognised or invalid DCompute targets: format is OpenCl "
        "(ocl-xy0 for x.y) CUDA (cuda-xy0 for cc x.y). Valid version"
        " for OpenCl are 100,110,120,200,210,220. Valid version for CUDA "
        "are 100,110,120,130,200,210,300,350,370,500,520,600,610,620");
  fatal();
  return nullptr;
}

DComputeCodeGenManager::DComputeCodeGenManager(llvm::LLVMContext &c) : ctx(c) {
  for (int i = 0; i < dcomputeTargets.size(); i++) {
    targets.push_back(createComputeTarget(dcomputeTargets[i]));
  }
}

void DComputeCodeGenManager::emit(Module *m) {
  for (int i = 0; i < targets.size(); i++) {
    gDComputeTarget = targets[i];
    targets[i]->emit(m);
    IrDsymbol::resetAll();
  }
}

void DComputeCodeGenManager::writeModules() {
  for (int i = 0; i < targets.size(); i++) {
    gDComputeTarget = targets[i];
    targets[i]->writeModule();
  }
}