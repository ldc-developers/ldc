//===-- targetCUDA.cpp ------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
#include "dcompute/target.h"
#include "dcompute/reflect.h"
#include "llvm/IR/metadata.h"
#include "dcompute/reflect.h"
#include "llvm/ADT/APint.h"
#include "dcompute/abi-cuda.h"
#include "gen/logger.h"

namespace {
class TargetCUDA : public DComputeTarget {
public:
  TargetCUDA(llvm::LLVMContext &c, int sm) : DComputeTarget(c,sm)
  {
      _ir = new IRState("dcomputeTargetCUDA",ctx);
      _ir->module.setTargetTriple( global.params.is64bit ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda");
 
    std::string dl;
    if (global.params.is64bit) {
        dl = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
    } else {
        dl = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
    }
    _ir->module.setDataLayout(dl);
    abi = createCudaABI();
    
    binSuffix= "ptx";
  }
  void runReflectPass() override {
    auto p = createDComputeReflectPass(2,tversion);
    p->runOnModule(_ir->module);
  }
 /* void runPointerReplacePass() override {
    //see http://llvm.org/docs/NVPTXUsage.html#address-spaces
    int mapping[PSnum] = {5, 1, 3, 4, 0,};
    auto p = createPointerReplacePass(mapping);
    p->runOnModule(*llm);
  }*/
  void addMetadata() override {
    //sm version?
  }
    
  void handleNonKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    
  }
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: Handle Function attibutes
    llvm::NamedMDNode *na = _ir->module.getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Metadata *fn    = llvm::ConstantAsMetadata::get(llf);
    llvm::Metadata *kstr  = llvm::MDString::get(ctx,"kernel");
    llvm::Metadata *one   = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx,32),1));

    llvm::Metadata *arr[] = {fn,kstr,one};
    llvm::MDNode *tup   = llvm::MDTuple::get(ctx,arr);
    na->addOperand(tup);

  }
    
};
}

DComputeTarget *createCUDATarget(llvm::LLVMContext& c, int sm) {
  return new TargetCUDA(c,sm);
};

