//===-- dcompute/codegenmanager.h - LDC command line options -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DCOMPUTE_CODEGENMANAGER_H
#define LDC_DCOMPUTE_CODEGENMANAGER_H

#include "dcompute/target.h"
#include <vector>

// gets run on modules makred @compute
// we do singleobj only
class DComputeCodeGenManager {
  
  struct target {
    int platform; //0 - host, 1 - OpenCL, 2 - CUDA
    int _version; //platform specific. OpenCL version we are pretending to be OR sm for CUDA
  };
  // targets from the command line. Hard coded for now. TODO: do this properly.
  // also SmallVector this
  static std::vector<target> clTargets;
  llvm::LLVMContext ctx;
  std::vector<DComputeTarget *> targets;
  DComputeTarget *createComputeTarget(target t);
public:
  void emit(Module *m);

    
  DComputeCodeGenManager();
  ~DComputeCodeGenManager();
    
};

#endif
