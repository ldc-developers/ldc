//===-- gen/dcomputetarget.h -------------------------------------*- C++-*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DCOMPUTETARGET_H
#define LDC_GEN_DCOMPUTETARGET_H
#include "gen/irstate.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
namespace llvm {
class Module;
class Function;
}
#define MAX_NUM_TARGET_ADDRSPACES 5
class Module;
class FuncDeclaration;

class DComputeTarget {
public:
  int tversion; // OpenCL or CUDA CC version:major*100 + minor*10
  int target;   // For cheap "dynamic casts" and ID for codegen time
                // conditional compilation. 0. host 1. OpenCL. 2. CUDA
  IRState *_ir;
  llvm::LLVMContext &ctx;
  TargetABI *abi;
  DComputeTarget(llvm::LLVMContext &c, int v);
  const char *binSuffix;
  void emit(Module *m);
  int mapping[MAX_NUM_TARGET_ADDRSPACES];
  void doCodeGen(Module *m);
  void writeModule();
  //HACK:Resets the gTargetMachine to one appropriate for this dcompute target
  virtual void setGTargetMachine() = 0;
  virtual void addMetadata() = 0;
  virtual void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) = 0;
};

DComputeTarget *createCUDATarget(llvm::LLVMContext &c, int sm);
DComputeTarget *createOCLTarget(llvm::LLVMContext &c, int oclver);

#endif
