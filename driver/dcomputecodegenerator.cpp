//===-- driver/dcomputecodegenerator.cpp ----------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/dcomputecodegenerator.h"
#include "driver/cl_options.h"
#include "driver/cl_helpers.h"
#include "dmd/errors.h"
#include "ir/irdsymbol.h"
#include "llvm/Support/CommandLine.h"
#include <array>
#include <string>
#include <algorithm>

#if !(LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX)

DComputeCodeGenManager::DComputeCodeGenManager(llvm::LLVMContext &c) : ctx(c) {}
void DComputeCodeGenManager::emit(Module *) {}
void DComputeCodeGenManager::writeModules() {}
DComputeCodeGenManager::~DComputeCodeGenManager() {}

#else

DComputeTarget *
DComputeCodeGenManager::createComputeTarget(const std::string &s) {
  if (s.substr(0, 4) == "ocl-") {
#if LDC_LLVM_SUPPORTED_TARGET_SPIRV
#define OCL_VALID_VER_INIT 100, 110, 120, 200, 210, 220
    const std::array<int, 6> valid_ocl_versions = {{OCL_VALID_VER_INIT}};

    const int v = atoi(s.c_str() + 4);
    if (std::find(valid_ocl_versions.begin(), valid_ocl_versions.end(), v) !=
        valid_ocl_versions.end()) {
      return createOCLTarget(ctx, v);
    }
#else
    error(Loc(), "LDC was not built with OpenCl DCompute support.");
#endif
  }

  if (s.substr(0, 5) == "cuda-") {
#if LDC_LLVM_SUPPORTED_TARGET_NVPTX
#define CUDA_VALID_VER_INIT 100, 110, 120, 130, 200, 210, 300, 350, 370,\
 500, 520, 600, 610, 620, 700, 720, 750, 800
    const std::array<int, 18> valid_cuda_versions = {{CUDA_VALID_VER_INIT}};

    const int v = atoi(s.c_str() + 5);
    if (std::find(valid_cuda_versions.begin(), valid_cuda_versions.end(), v) !=
        valid_cuda_versions.end()) {
      return createCUDATarget(ctx, v);
    }
#else
    error(Loc(), "LDC was not built with CUDA DCompute support.");
#endif
  }

#define STR(...) #__VA_ARGS__
#define XSTR(x) STR(x)

  error(Loc(),
        "Unrecognised or invalid DCompute targets: the format is ocl-xy0 "
        "for OpenCl x.y and cuda-xy0 for CUDA CC x.y."
#if LDC_LLVM_SUPPORTED_TARGET_SPIRV
        " Valid version strings for OpenCl are ocl-{" XSTR(OCL_VALID_VER_INIT) "}."
#endif
#if LDC_LLVM_SUPPORTED_TARGET_NVPTX
        " Valid version strings for CUDA are cuda-{" XSTR(CUDA_VALID_VER_INIT) "}."
#endif
  );

#undef XSTR
#undef STR

  fatal();
  return nullptr;
}

DComputeCodeGenManager::DComputeCodeGenManager(llvm::LLVMContext &c) : ctx(c) {
  for (auto &option : opts::dcomputeTargets) {
    targets.push_back(createComputeTarget(option));
  }
  oldGIR = gIR;
  oldGTargetMachine = gTargetMachine;
}

void DComputeCodeGenManager::emit(Module *m) {
  for (auto &target : targets) {
    target->emit(m);
    IrDsymbol::resetAll();
  }
}

void DComputeCodeGenManager::writeModules() {
  for (auto &target : targets) {
    target->writeModule();
  }
}

DComputeCodeGenManager::~DComputeCodeGenManager() {
  gIR = oldGIR;
  gTargetMachine = oldGTargetMachine;
}

#endif // LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX
