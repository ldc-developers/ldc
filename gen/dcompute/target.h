//===-- gen/dcompute/target.h ------------------------------------*- C++-*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DCOMPUTE_TARGET_H
#define LDC_GEN_DCOMPUTE_TARGET_H
#include "gen/irstate.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include <array>
namespace llvm {
class Module;
class Function;
}

class Module;
class FuncDeclaration;

class DComputeTarget {
public:
  llvm::LLVMContext &ctx;
  int tversion; // OpenCL or CUDA CC version:major*100 + minor*10
  enum ID { Host = 0, OpenCL = 1, CUDA = 2 };
  ID target;    // ID for codegen time conditional compilation.
  const char *short_name;
  const char *binSuffix;
  TargetABI *abi;
  // The nominal address spaces in DCompute are Private = 0, Global = 1,
  // Shared = 2, Constant = 3, Generic = 4
  std::array<int, 5> mapping;

  IRState *_ir;

  DComputeTarget(llvm::LLVMContext &c, int v, ID id, const char *_short_name,
                 const char *suffix, TargetABI *a, std::array<int, 5> map)
      : ctx(c), tversion(v), target(id), short_name(_short_name),
        binSuffix(suffix), abi(a), mapping(map), _ir(nullptr) {}

  void emit(Module *m);
  void doCodeGen(Module *m);
  void writeModule();

  // HACK: Resets the gTargetMachine to one appropriate for this dcompute target
  virtual void setGTargetMachine() = 0;
  virtual void addMetadata() = 0;
  virtual void addKernelMetadata(FuncDeclaration *df, llvm::Function *llf) = 0;
};

DComputeTarget *createCUDATarget(llvm::LLVMContext &c, int sm);
DComputeTarget *createOCLTarget(llvm::LLVMContext &c, int oclver);

#endif
