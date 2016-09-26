//===-- dcompute/codegenmanager.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/codegenmanager.h"
#include "ir/irdsymbol.h"
#include "driver/cl_options.h"
#include "gen/cl_helpers.h"
#include "llvm/Support/CommandLine.h"
#include "ddmd/errors.h"
#include <string>
namespace cl = llvm::cl;
static cl::list<std::string>
    mDcomputeTargets("mdcompute-targets", cl::CommaSeparated,
                     cl::desc("DCompute targets to generate for:OpenCl "
                              "(ocl-xy0 for x.y) CUDA (cuda-xy0 for cc x.y)"),
                     cl::value_desc("ocl-210,cuda-350"));

DComputeTarget *
DComputeCodeGenManager::createComputeTarget(const std::string &s) {
  int v;
  if (s.substr(0, 4) == "ocl-") {
    v = atoi(s.c_str() + 4);
    if (v==100||v==110||v==120||v==200||v==210||v==220) {
      return createOCLTarget(ctx, v);
    }
  } else if (s.substr(0, 5) == "cuda-") {
    //TODO: validate version
    v = atoi(s.c_str() + 5);
    return createCUDATarget(ctx, v);
  }
  error(Loc(),"unrecognised or invalid DCompute targets: format is OpenCl "
              "(ocl-xy0 for x.y) CUDA (cuda-xy0 for cc x.y)");
  fatal();
  return nullptr;
}

DComputeCodeGenManager::DComputeCodeGenManager(llvm::LLVMContext &c) : ctx(c) {
  for (int i = 0; i < mDcomputeTargets.size(); i++) {
    targets.push_back(createComputeTarget(mDcomputeTargets[i]));
  }
}

void DComputeCodeGenManager::emit(Module *m) {
  for (int i = 0; i < targets.size(); i++) {
    gDComputeTarget = targets[i];
    targets[i]->emit(m);
    IrDsymbol::resetAll();
  }
}

DComputeCodeGenManager::~DComputeCodeGenManager() {}

void DComputeCodeGenManager::writeModules() {
  for (int i = 0; i < targets.size(); i++) {
    gDComputeTarget = targets[i];
    targets[i]->writeModule();
  }
}
