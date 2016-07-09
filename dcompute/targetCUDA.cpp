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
namespace {
class TargetCUDA : DComputeTarget {
  int sm_target;
  TargetCUDA(Module *_m, int sm) : DComputeTarget(_m), sm_target(sm)
  {}
  void runReflectPass() override {
    auto p = createReflectPass(2,sm_target);
    p->runOnModule(*llm);
  }
  void runPointerReplacePass() override {
    //see http://llvm.org/docs/NVPTXUsage.html#address-spaces
    int mapping[PSnum] = {5, 1, 3, 4, 0,};
    auto p = createPointerReplacePass(mapping);
    p->runOnModule(*llm);
  }
  void addMetadata() override {
    //sm version?
  }
    
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: set the calling convention for llf to llvm::CallingConv::PTX_DEVICE
  }
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: Handle Function attibutes
    llvm::NamedMDNode *na = llm->getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Metadata *fn    = llvm::ConstantAsMetadata::get(llf);
    llvm::Metadata *kstr  = llvm::MDString::get(getGblobalContext(),"kernel");
    llvm::Metadata *one   = llvm::ConstantAsMetadata::get(IntegerType::getInt32Ty(),1,false);
    llvm::Metadata *arr[] = {fn,kstr,one};
    llvm::Metadata *tup   = llvm::MDTuple::get(arr);
    na->addOperand(tup);
  }
    
}
}

DComputeTarget *createCUDATarget(Module *_m, int sm) {
  return new TargetCUDA(_m,sm);
};

