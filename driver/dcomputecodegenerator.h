//===-- driver/dcomputecodegenerator.h - LDC --------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/dcompute/target.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  class TargetMachine;
}

// gets run on modules marked @compute
// All @compute D modules are emitted into one LLVM module once per target.
class DComputeCodeGenManager {

  llvm::LLVMContext &ctx;
  llvm::SmallVector<DComputeTarget *, 2> targets;
  DComputeTarget *createComputeTarget(const std::string &s);
  IRState *oldGIR = nullptr;
  llvm::TargetMachine *oldGTargetMachine = nullptr;
public:
  void emit(Module *m);
  void writeModules();

  DComputeCodeGenManager(llvm::LLVMContext &c);
  ~DComputeCodeGenManager();
};
