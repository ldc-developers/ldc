//===-- target.h ------------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#ifndef LDC_DCOMPUTE_TARGET_H
#define LDC_DCOMPUTE_TARGET_H
namespace llvm {
  class Module;
  class Function;
}

class Module;
class FuncDeclaration;

namespace ldc {
class DComputeTarget {
  llvm::Module *llm;
  Module *dm;
    
    
public:
  DComputeTarget(Module* m) : m(m) llm(nullptr){}
  void doCodeGen();
  virtual void runReflectPass();
  virtual void runPointerReplacePass();
  virtual void runSpecialTypeReplacePass();
  virtual void addMetadata();
  virtual void handleKernelFunc(FuncDeclaration *df, llvm::Funtion *llf);
  virtual void handleNonKernelFunc(FuncDeclaration *df, llvm::Funtion *llf);
};
}
DComputeTarget *createCUDATarget(Module *_m, int sm);
DComputeTarget *createOCLTarget(Module *_m, int oclver);

#endif
