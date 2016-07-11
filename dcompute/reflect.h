//===-- dcompute/reflect.h - LDC dcompute reflace pass ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DCOMPUTE_REFLECT_H
#define LDC_DCOMPUTE_REFLECT_H

namespace llvm {
  class ModulePass;
}
enum PointerSpace {
    PSPrivate = 0,
    PSGlobal = 1,
    PSShared = 2,
    PSConstant = 3,
    PSGeneric = 4,
    PSnum = 5
    };
llvm::ModulePass *createDComputeReflectPass(int, unsigned);

#endif
