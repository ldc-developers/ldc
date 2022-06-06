//===-- gen/abi-mips64.cpp - MIPS64 ABI description ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The MIPS64 N32 and N64 ABI can be found here:
// http://techpubs.sgi.com/library/dynaweb_docs/0640/SGI_Developer/books/Mpro_n32_ABI/sgi_html/index.html
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-mips64.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct MIPS64TargetABI : TargetABI {
  const bool Is64Bit;

  explicit MIPS64TargetABI(const bool Is64Bit) : Is64Bit(Is64Bit) {}

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref()) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    // Return structs and static arrays on the stack. The latter is needed
    // because otherwise LLVM tries to actually return the array in a number
    // of physical registers, which leads, depending on the target, to
    // either horrendous codegen or backend crashes.
    return (rt->ty == TY::Tstruct || rt->ty == TY::Tsarray);
  }

  bool passByVal(TypeFunction *, Type *t) override {
    TY ty = t->toBasetype()->ty;
    return ty == TY::Tstruct || ty == TY::Tsarray;
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
    // FIXME
  }
};

// The public getter for abi.cpp
TargetABI *getMIPS64TargetABI(bool Is64Bit) {
  return new MIPS64TargetABI(Is64Bit);
}
