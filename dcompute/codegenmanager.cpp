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
#include <string>
namespace cl = llvm::cl;
static cl::list<std::string>
    mDcomputeTargets("mdct", cl::CommaSeparated,
                     cl::desc("DCompute targets to generate for:OpenCl "
                              "(ocl-xy0 for x.y) CUDA (cuda-xy0 for cc x.y)"),
                     cl::value_desc("ocl-210,cuda-350"));
std::vector<DComputeCodeGenManager::target> DComputeCodeGenManager::clTargets =
    {{1, 210}, {2, 350}};
DComputeTarget *
DComputeCodeGenManager::createComputeTarget(const std::string &s) {
  int v;
  if (s.substr(0, 3) == "ocl") {
    v = atoi(s.c_str() + 4);
    return createOCLTarget(ctx, v);
  } else if (s.substr(0, 4) == "cuda") {
    v = atoi(s.c_str() + 5);
    return createCUDATarget(ctx, v);
  }
  // TODO: print an error msg
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
