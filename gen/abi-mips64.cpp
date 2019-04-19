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
  IndirectByvalRewrite byvalRewrite;

  explicit MIPS64TargetABI(const bool Is64Bit) : Is64Bit(Is64Bit) {}

  bool passPointerToHiddenCopy(Type *t, bool isReturnValue) const {
    if (isReturnValue && !isPOD(t, false))
        return true;
    // Remaining aggregates which can NOT be rewritten as integers (size > 8
    // bytes or not a power of 2) are passed by ref to hidden copy.
    return isAggregate(t) && !canRewriteAsInt(t);
  }

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref) {
      return false;
    }
    Type *rt = tf->next->toBasetype();
    return passPointerToHiddenCopy(rt, /*isReturnValue=*/true);
  }
  // This was disabled as `interface` calls where segfaulting when
  // using large aggregate parameters. See LDC issue #3050.
  // Alternatively, the usage of `byvalRewrite` will solve the issue in a hackish way.
  // ATM this ABI is not C compatible.
  bool passByVal(TypeFunction *, Type *t) override {
    return false;
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    const auto rt = fty.ret->type->toBasetype();
    if (!fty.ret->byref && rt->ty != Tvoid) {
      rewrite(fty, *fty.ret, /*isReturnValue=*/true);
    }

    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
      rewrite(fty, arg, /*isReturnValue=*/false);
  }

  void rewrite(IrFuncTy &fty, IrFuncTyArg &arg, bool isReturnValue) {
    Type *t = arg.type->toBasetype();

    if (passPointerToHiddenCopy(t, isReturnValue)) {
      // the caller allocates a hidden copy and passes a pointer to that copy
      byvalRewrite.applyTo(arg);
    }

    if (arg.rewrite) {
      LLType *originalLType = arg.ltype;
      IF_LOG {
        Logger::println("Rewriting argument type %s", t->toChars());
        LOG_SCOPE;
        Logger::cout() << *originalLType << " => " << *arg.ltype << '\n';
      }
    }
  }
  };

// The public getter for abi.cpp
TargetABI *getMIPS64TargetABI(bool Is64Bit) {
  return new MIPS64TargetABI(Is64Bit);
}
