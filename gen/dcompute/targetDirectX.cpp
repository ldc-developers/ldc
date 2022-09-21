//===-- gen/dcompute/targetDirectX.cpp ------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// Parts of this file are adapted from CodeGenFunction.cpp (Clang, LLVM).
// Therefore, this file is distributed under the LLVM license.
// See the LICENSE file for details.
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_DirectX

#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/template.h"
#include "dmd/module.h"
#include "gen/abi-targets.h"
#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/logger.h"
#include "llvm/Transforms/Scalar.h"
#include "gen/optimizer.h"
#include "driver/targetmachine.h"
#include <cstring>
#include <string>

namespace {
class TargetDirectX : public DComputeTarget {
public:
    TargetDirectX(llvm::LLVMContext &c, int shadermodel)
      : DComputeTarget(c, shadermodel, ID::DirectX, "directx", "dxil",
                       createDirectXABI(),
                       {{0, 0, 0, 0, 0}}) {

    _ir = new IRState("dcomputeTargetDirectX", ctx);
          
    // TODO: modes other then just library,
    /*
     .Case("ps",  Triple::EnvironmentType::Pixel)
     .Case("vs",  Triple::EnvironmentType::Vertex)
     .Case("gs",  Triple::EnvironmentType::Geometry)
     .Case("hs",  Triple::EnvironmentType::Hull)
     .Case("ds",  Triple::EnvironmentType::Domain)
     .Case("cs",  Triple::EnvironmentType::Compute)
     .Case("lib", Triple::EnvironmentType::Library)
     .Case("ms",  Triple::EnvironmentType::Mesh)
     .Case("as",  Triple::EnvironmentType::Amplification)
     */
    //TODO: (command line option for) shadermodel version
    // do we care about anything other than `pc` here?
          
    std::string tripleString = "dxil-pc-shadermodel6.3-compute";
    auto floatABI = ::FloatABI::Hard;
    _ir->module.setTargetTriple(tripleString);
     targetMachine = createTargetMachine(
              tripleString, "dxil",
              "", {},
              ExplicitBitness::M32, floatABI,
              llvm::Reloc::Static, llvm::CodeModel::Medium, codeGenOptLevel(), false);
    // Copied from
    _ir->module.setDataLayout(
        "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64");
    _ir->dcomputetarget = this;
  }

  void addMetadata() override {
    /* generate
     !llvm.module.flags = !{!}
     !dx.valver = !{!1}
     !1 = !{i32 1, i32 1}
     */
    llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
    auto i32_1 = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1));
    auto i32_7 = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 7));
    llvm::Metadata *BangZero[] = { i32_1, i32_7 };
    llvm::NamedMDNode *dx_valver = _ir->module.getOrInsertNamedMetadata("dx.valver");
    dx_valver->addOperand(llvm::MDNode::get(ctx, BangZero));

    /*llvm::NamedMDNode *LLVMModFlags =
        _ir->module.getOrInsertNamedMetadata("llvm.module.flags");
    LLVMModFlags->addOperand(llvm::MDNode::get(ctx, BangZero));*/
  }
  
  void addKernelMetadata(FuncDeclaration *fd, llvm::Function *llf) override {
    //FIXME: is this needed if the triple's enviroment type is specified?
    llf->addFnAttr("hlsl.shader", "compute");
    //TODO: numthreads
  }
  
};
} // anonymous namespace.

DComputeTarget *createDirectXTarget(llvm::LLVMContext &c, int dxver) {
  return new TargetDirectX(c, dxver);
}

#endif // LDC_LLVM_SUPPORTED_TARGET_DIRECTX
