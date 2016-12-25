//===-- dcompute/codegenmanager.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/dcomputecodegenmanager.h"
#include "driver/cl_options.h"
#include "ddmd/errors.h"
#include "gen/cl_helpers.h"
#include "ir/irdsymbol.h"
#include "llvm/Support/CommandLine.h"
#include <string>
#include <algorithm>

DComputeTarget *
DComputeCodeGenManager::createComputeTarget(const std::string &s) {
  int v;
#define OCL_VALID_VER_INIT 100, 110, 120, 200, 210, 220
  const llvm::SmallVector<int, 6> valid_ocl_versions = {OCL_VALID_VER_INIT};
#define CUDA_VALID_VER_INIT 100, 110, 120, 130, 200, 210, 300, 350, 370,\
 500, 520, 600, 610, 620
  const llvm::SmallVector<int, 14> vaild_cuda_versions = {CUDA_VALID_VER_INIT};

  if (s.substr(0, 4) == "ocl-") {
    v = atoi(s.c_str() + 4);
    if (find(valid_ocl_versions.begin(), valid_ocl_versions.end(), v) !=
        valid_ocl_versions.end()) {
      return createOCLTarget(ctx, v);
    }
  } else if (s.substr(0, 5) == "cuda-") {
    v = atoi(s.c_str() + 5);

    if (find(vaild_cuda_versions.begin(), vaild_cuda_versions.end(), v) !=
        vaild_cuda_versions.end()) {
      return createOCLTarget(ctx, v);
    }
  }
#define STR(x) #x
  error(Loc(),
        "unrecognised or invalid DCompute targets: the format is ocl-xy0 "
        "for OpenCl x.y and cuda-xy0 for CUDA CC x.y. Valid versions "
        "for OpenCl are " STR(OCL_VALID_VER_INIT) ". Valid versions for CUDA "
        "are " STR(CUDA_VALID_VER_INIT));
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
    targets[i]->emit(m);
    IrDsymbol::resetAll();
  }
}

void DComputeCodeGenManager::writeModules() {
  for (int i = 0; i < targets.size(); i++) {
    targets[i]->writeModule();
  }
}
