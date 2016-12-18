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
  int tversion;
  int target;
  IRState *_ir;
  llvm::LLVMContext &ctx;
  TargetABI *abi;
  DComputeTarget(llvm::LLVMContext &c, int v);
  const char *binSuffix;
  void emit(Module *m);
  int mapping[MAX_NUM_TARGET_ADDRSPACES];
  void doCodeGen(Module *m);
  void writeModule();
  virtual void setGTargetMachine() = 0;
  virtual void addMetadata() = 0;
  virtual void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) = 0;
};

DComputeTarget *createCUDATarget(llvm::LLVMContext &c, int sm);
DComputeTarget *createOCLTarget(llvm::LLVMContext &c, int oclver);

#endif