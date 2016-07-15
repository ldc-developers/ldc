//===-- dcompute/target.h ----------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#ifndef LDC_DCOMPUTE_TARGET_H
#define LDC_DCOMPUTE_TARGET_H
#include "gen/irstate.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "dcompute/reflect.h"
namespace llvm {
  class Module;
  class Function;
}

class Module;
class FuncDeclaration;

class DComputeTarget {
public:
  int tversion;
  IRState* _ir;
  llvm::LLVMContext &ctx;
  TargetABI* abi;
  DComputeTarget(llvm::LLVMContext &c, int v);
  char *  binSuffix;
  void emit(Module* m);
  int mapping[PSnum];
  void doCodeGen(Module* m);
  void writeModule();
  virtual void runReflectPass() =0;
  //virtual void runPointerReplacePass();
  //virtual void runSpecialTypeReplacePass();
  virtual void addMetadata() =0;
  virtual void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) =0;
  virtual void handleNonKernelFunc(FuncDeclaration *df, llvm::Function *llf)=0;
};

DComputeTarget *createCUDATarget(llvm::LLVMContext &c, int sm);
DComputeTarget *createOCLTarget(llvm::LLVMContext &c, int oclver);

#endif
