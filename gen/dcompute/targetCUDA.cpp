//===-- gen/dcompute/targetCUDA.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_NVPTX

#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/passes/metadata.h"
#include "gen/abi-nvptx.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/to_string.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "driver/targetmachine.h"
#include <cstring>

namespace {
class TargetCUDA : public DComputeTarget {
public:
  TargetCUDA(llvm::LLVMContext &c, int sm)
      : DComputeTarget(
            c, sm, CUDA, "cuda", "ptx", createNVPTXABI(),

            // Map from nominal DCompute address space to NVPTX address space.
            // see $LLVM_ROOT/docs/docs/NVPTXUsage.rst section Address Spaces
            {{5, 1, 3, 4, 0}}) {

    const bool is64 = global.params.targetTriple->isArch64Bit();
    auto tripleString = is64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda";

    auto floatABI = ::FloatABI::Hard;
    targetMachine = createTargetMachine(
        tripleString, is64 ? "nvptx64" : "nvptx",
        "sm_" + ldc::to_string(tversion / 10), {},
        is64 ? ExplicitBitness::M64 : ExplicitBitness::M32, floatABI,
        llvm::Reloc::Static, llvm::CodeModel::Medium, codeGenOptLevel(), false);

    _ir = new IRState("dcomputeTargetCUDA", ctx);
    _ir->module.setTargetTriple(tripleString);
    _ir->module.setDataLayout(targetMachine->createDataLayout());
    _ir->dcomputetarget = this;
  }

  void addMetadata() override {
    // sm version?
  }

  void addKernelMetadata(FuncDeclaration *df, llvm::Function *llf) override {
    // TODO: Handle Function attibutes
    llvm::NamedMDNode *na =
        _ir->module.getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Metadata *fn = llvm::ConstantAsMetadata::get(llf);
    llvm::Metadata *kstr = llvm::MDString::get(ctx, "kernel");
    llvm::Metadata *one = llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 1));

    llvm::Metadata *arr[] = {fn, kstr, one};
    llvm::MDNode *tup = llvm::MDTuple::get(ctx, arr);
    na->addOperand(tup);
  }
};
} // anonymous namespace.

DComputeTarget *createCUDATarget(llvm::LLVMContext &c, int sm) {
  return new TargetCUDA(c, sm);
};

#endif // LDC_LLVM_SUPPORTED_TARGET_NVPTX
