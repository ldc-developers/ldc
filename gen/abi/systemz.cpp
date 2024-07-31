//===-- abi-systemz.cpp
//-----------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit big-endian IBM Z targets.
//
// The IBM s390x ELF ABI can be found here:
// https://github.com/IBM/s390x-abi
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct SystemZTargetABI : TargetABI {
  IndirectByvalRewrite indirectByvalRewrite{};

  explicit SystemZTargetABI() {}

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref()) {
      return false;
    }
    Type *rt = tf->next->toBasetype();
    return DtoIsInMemoryOnly(rt);
  }

  bool passByVal(TypeFunction *, Type *t) override {
    return DtoIsInMemoryOnly(t);
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
    if (!isPOD(arg.type)) {
      // non-PODs should be passed in memory
      indirectByvalRewrite.applyTo(arg);
      return;
    }
    Type *ty = arg.type->toBasetype();
    // integer types less than 64-bits should be extended to 64 bits
    if (ty->isintegral()) {
      arg.attrs.addAttribute(ty->isunsigned() ? LLAttribute::ZExt
                                              : LLAttribute::SExt);
    }
  }
};

// The public getter for abi.cpp
TargetABI *getSystemZTargetABI() { return new SystemZTargetABI(); }
