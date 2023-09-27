//===-- gen/abi-loongarch64.cpp - LoongArch64 ABI description -----------*- C++
//-*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ABI spec:
// https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html
//
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct LoongArch64TargetABI : TargetABI {
private:
  IndirectByvalRewrite indirectByvalRewrite{};

public:
  auto returnInArg(TypeFunction *tf, bool) -> bool override {
    if (tf->isref()) {
      return false;
    }
    Type *rt = tf->next->toBasetype();
    if (!isPOD(rt)) {
      return true;
    }
    // pass by reference when > 2*GRLEN
    return rt->size() > 16;
  }

  auto passByVal(TypeFunction *, Type *t) -> bool override {
    if (!isPOD(t)) {
      return false;
    }
    return t->size() > 16;
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!fty.ret->byref) {
      rewriteArgument(fty, *fty.ret);
    }

    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    if (arg.byref) {
      return;
    }

    if (!isPOD(arg.type)) {
      // non-PODs should be passed in memory
      indirectByvalRewrite.applyTo(arg);
      return;
    }
  }
};

// The public getter for abi.cpp
TargetABI *getLoongArch64TargetABI() { return new LoongArch64TargetABI(); }
