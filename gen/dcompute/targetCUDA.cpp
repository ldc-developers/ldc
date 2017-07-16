//===-- gen/dcompute/targetCUDA.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/metadata.h"
#include "gen/abi-nvptx.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
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
    std::string dl =
        global.params.is64bit
            ? "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:"
              "32-"
              "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
            : "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:"
              "32-"
              "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

    _ir = new IRState("dcomputeTargetCUDA", ctx);
    _ir->module.setTargetTriple(global.params.is64bit ? "nvptx64-nvidia-cuda"
                                                      : "nvptx-nvidia-cuda");
    _ir->module.setDataLayout(dl);
    _ir->dcomputetarget = this;
  }

  void addMetadata() override {
    // sm version?
  }
  void setGTargetMachine() override {
    char buf[8];
    bool is64 = global.params.is64bit;
    snprintf(buf, sizeof(buf), "sm_%d", tversion / 10);
    gTargetMachine = createTargetMachine(
        is64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda",
        is64 ? "nvptx64" : "nvptx", buf, {},
        is64 ? ExplicitBitness::M64 : ExplicitBitness::M32, ::FloatABI::Hard,
        llvm::Reloc::Static, llvm::CodeModel::Medium, codeGenOptLevel(), false,
        false);
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
