//===-- driver/dcomputecodegenerator.h - LDC --------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_DCOMPUTECODEGENERATOR_H
#define LDC_DRIVER_DCOMPUTECODEGENERATOR_H

#include "gen/dcomputetarget.h"
#include "llvm/ADT/SmallVector.h"

// gets run on modules marked @compute
// we do singleobj only
class DComputeCodeGenManager {

  struct target {
    int platform; // 0 - host, 1 - OpenCL, 2 - CUDA
    int _version; // platform specific. OpenCL version we are pretending to be
                  // OR sm for CUDA
  };

  llvm::LLVMContext &ctx;
  llvm::SmallVector<DComputeTarget *, 2> targets;
  DComputeTarget *createComputeTarget(const std::string &s);

public:
  void emit(Module *m);
  void writeModules();

  DComputeCodeGenManager(llvm::LLVMContext &c);
};

#endif