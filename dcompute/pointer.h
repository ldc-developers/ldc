//===-- dcompute/pointer.h - LDC dcompute pointer replacement pass -*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
#ifndef LDC_DCOMPUTE_POINTER_H
#define LDC_DCOMPUTE_POINTER_H
//N.B this order MUST macth dcompute.types.pointer
enum pointerspace {
  PSPrivate  = 0,
  PSGlobal   = 1,
  PSShared   = 2,
  PSConstant = 3,
  PSGeneric  = 4,
  PSnum      = 5
};
namespace llvm {
    class ModulePass;
}
llvm::ModulePass *createPointerReplacePass(int mapping[PSnum]);

#endif
