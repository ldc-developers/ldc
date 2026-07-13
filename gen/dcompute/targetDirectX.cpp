//===-- gen/dcompute/targetDirectX.cpp ------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// DirectX / DXIL dcompute backend scaffold.
// Metadata style mirrors Vulkan (HLSL function attrs) + DXIL compute triple,
// per Clang CodeGenHLSL / llvm/test/CodeGen/DirectX requirements.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_DirectX

#include "dmd/expression.h"
#include "gen/abi/targets.h"
#include "gen/dcompute/druntime.h"
#include "gen/dcompute/target.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/to_string.h"
#include "driver/targetmachine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Target/TargetMachine.h"
#include <string>

using namespace dmd;

namespace {

class TargetDirectX : public DComputeTarget {
public:
  TargetDirectX(llvm::LLVMContext &c, int smVersion)
      : DComputeTarget(c, smVersion, ID::DirectX, "directx", "ll",
                       createDirectXABI(),
                       // Private, Global, Shared, Constant, Generic
                       // Shared → addrspace(3) (Clang group_shared / DXIL)
                       {{0, 1, 3, 2, 0}}) {

    const int major = tversion / 100;
    const int minor = (tversion % 100) / 10;
    std::string tripleString =
        "dxil-pc-shadermodel" + ldc::to_string(major) + "." +
        ldc::to_string(minor) + "-compute";

    auto floatABI = ::FloatABI::Hard;
    targetMachine = createTargetMachine(
        tripleString, "dxil", "", {}, ExplicitBitness::None, floatABI,
        llvm::Reloc::Static, llvm::CodeModel::Medium, codeGenOptLevel(), false);

    _ir = new IRState("dcomputeTargetDirectX", ctx);
#if LLVM_VERSION_MAJOR >= 21
    _ir->module.setTargetTriple(llvm::Triple(tripleString));
#else
    _ir->module.setTargetTriple(tripleString);
#endif
    _ir->module.setDataLayout(targetMachine->createDataLayout());
    _ir->dcomputetarget = this;
  }

  void addMetadata() override {
    // Module-level MD left empty initially (same as CUDA/Vulkan).
    // dxil-translate-metadata derives !dx.shaderModel from triple + fn attrs.
  }

  llvm::AttrBuilder buildKernAttrs(StructLiteralExp *kernAttr) {
    auto b = llvm::AttrBuilder(ctx);
    b.addAttribute("hlsl.shader", "compute");

    // @kernel(size_t[3] bounds) — first struct field is the array literal.
    std::string numthreads = "1,1,1";
    if (kernAttr && kernAttr->elements && kernAttr->elements->length > 0) {
      if (auto *ale = (*kernAttr->elements)[0]->isArrayLiteralExp()) {
        if (ale->elements && ale->elements->length >= 3) {
          numthreads.clear();
          numthreads += ldc::to_string((*ale->elements)[0]->toInteger());
          numthreads += ",";
          numthreads += ldc::to_string((*ale->elements)[1]->toInteger());
          numthreads += ",";
          numthreads += ldc::to_string((*ale->elements)[2]->toInteger());
        }
      }
    }
    b.addAttribute("hlsl.numthreads", numthreads);

    // Present in LLVM DirectX fixtures before dxil-prepare; harmless for IR
    // inspection. Passes may strip it later.
    b.addAttribute("exp-shader", "cs");
    return b;
  }

  void addKernelMetadata(FuncDeclaration *df, llvm::Function *llf,
                         StructLiteralExp *kernAttr) override {
    (void)df;
    // Same HLSL-shaped attrs as Vulkan's buildKernAttrs — required by DXIL.
    llf->addFnAttrs(buildKernAttrs(kernAttr));
  }
};

} // namespace

DComputeTarget *createDirectXTarget(llvm::LLVMContext &c, int smVersion) {
  return new TargetDirectX(c, smVersion);
}

#endif // LDC_LLVM_SUPPORTED_TARGET_DirectX
