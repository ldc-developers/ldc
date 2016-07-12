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

namespace {
class TargetCUDA : public DComputeTarget {
public:
  TargetCUDA(llvm::LLVMContext &c, int sm) : DComputeTarget(c,sm)
  {
      _ir = new IRState("dcomputeTargetCUDA",ctx);
      _ir->module.setTargetTriple( global.params.is64bit ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda");
      //TODO: does this need to be changed
#if LDC_LLVM_VER >= 308
      _ir->module.setDataLayout(*gDataLayout);
#else
      _ir->module.setDataLayout(gDataLayout->getStringRepresentation());
#endif
      abi = createCudaABI();
      //Dont need the diBuilder to run
      //IrDsymbol::resetAll(); //this doesn't look good
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

