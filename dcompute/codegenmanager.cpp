//===-- codegenmanager.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/codegenmanager.h"


DComputeTarget *DComputeCodeGenManager::createComputeTarget(target t) {
  switch (t.platform) {
    case 1:
      return createOCLTarget(m,t._version);
      break;
    case 2:
      return createCUDATarget(m,t._version);
      break;
    default:
      llvm_unreachable("no such compute target");

  }
  
}

DComputeCodeGenManager::DComputeCodeGenManager(Module *_m) : m(_m) {
  for (int i = 0; i < clTargets.length(); i++) {
    dcTargets.push_back(createComputeTarget(clTargets[i]));
  }
}

