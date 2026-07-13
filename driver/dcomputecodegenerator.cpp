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

#if !(LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX ||     \
      LDC_LLVM_SUPPORTED_TARGET_DirectX)

DComputeCodeGenManager::DComputeCodeGenManager(llvm::LLVMContext &c) : ctx(c) {}
void DComputeCodeGenManager::emit(Module *) {}
void DComputeCodeGenManager::writeModules(llvm::Module *) {}
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

  if (s.substr(0, 8) == "directx-") {
#if LDC_LLVM_SUPPORTED_TARGET_DirectX
    // Shader Model x.y encoded as x*100 + y*10 (e.g. directx-660 → SM 6.6)
#define DIRECTX_VALID_VER_INIT 600, 610, 620, 630, 640, 650, 660, 670
    const std::array<int, 8> valid_dx_versions = {{DIRECTX_VALID_VER_INIT}};

    const int v = atoi(s.c_str() + 8);
    if (std::find(valid_dx_versions.begin(), valid_dx_versions.end(), v) !=
        valid_dx_versions.end()) {
      return createDirectXTarget(ctx, v);
    }
#else
    error(Loc(), "LDC was not built with DirectX DCompute support.");
#endif
  }

#define STR(...) #__VA_ARGS__
#define XSTR(x) STR(x)

  error(Loc(),
        "Unrecognised or invalid DCompute targets: the format is ocl-xy0 "
        "for OpenCl x.y, cuda-xy0 for CUDA CC x.y, and directx-xy0 for "
        "DirectX Shader Model x.y."
#if LDC_LLVM_SUPPORTED_TARGET_SPIRV
        " Valid version strings for OpenCl are ocl-{" XSTR(OCL_VALID_VER_INIT) "}."
#endif
#if LDC_LLVM_SUPPORTED_TARGET_NVPTX
        " Valid version strings for CUDA are cuda-{" XSTR(CUDA_VALID_VER_INIT) "}."
#endif
#if LDC_LLVM_SUPPORTED_TARGET_DirectX
        " Valid version strings for DirectX are directx-{" XSTR(DIRECTX_VALID_VER_INIT) "}."
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
  oldGABI = gABI;
}

void DComputeCodeGenManager::emit(Module *m) {
  for (auto &target : targets) {
    target->emit(m);
    IrDsymbol::resetAll();
  }
  gIR = oldGIR;
  gTargetMachine = oldGTargetMachine;
  gABI = oldGABI;
}

void DComputeCodeGenManager::writeModules(llvm::Module *hostModule) {
  for (auto &target : targets) {
    // Set target machine before writing the module, since writeModule uses it
    gTargetMachine = target->targetMachine;
    target->writeModule(hostModule);
  }
  gIR = oldGIR;
  gTargetMachine = oldGTargetMachine;
  gABI = oldGABI;
}

DComputeCodeGenManager::~DComputeCodeGenManager() {
  for (auto t : targets) {
    delete t;
  }
  gIR = oldGIR;
  gTargetMachine = oldGTargetMachine;
  gABI = oldGABI;
}

#endif // SPIRV || NVPTX || DirectX
