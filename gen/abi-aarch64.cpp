//===-- abi-aarch64.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The Procedure Call Standard can be found here:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
// https://github.com/ARM-software/software-standards/blob/master/abi/aapcs64/aapcs64.rst
//
//===----------------------------------------------------------------------===//

#include "gen/abi-aarch64.h"

#include "dmd/identifier.h"
#include "dmd/ldcbindings.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"

/**
 * The AAPCS64 uses a special native va_list type:
 *
 * typedef struct __va_list {
 *     void *__stack; // next stack param
 *     void *__gr_top; // end of GP arg reg save area
 *     void *__vr_top; // end of FP/SIMD arg reg save area
 *     int __gr_offs; // offset from __gr_top to next GP register arg
 *     int __vr_offs; // offset from __vr_top to next FP/SIMD register arg
 * } va_list;
 *
 * In druntime, the struct is defined as object.__va_list, an alias of
 * ldc.internal.vararg.std.__va_list.
 * Arguments of this type are never passed by value, only by reference (even
 * though the mangled function name indicates otherwise!). This requires a
 * little bit of compiler magic in the following implementations.
 */
struct AArch64TargetABI : TargetABI {
private:
  const bool isDarwin;
  IndirectByvalRewrite byvalRewrite;
  HFVAToArray hfvaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  bool isAAPCS64VaList(Type *t) {
    return !isDarwin && t->ty == Tstruct &&
           strcmp(t->toPrettyChars(true),
                  "ldc.internal.vararg.std.__va_list") == 0;
  }

  bool passIndirectlyByValue(Type *t) {
    t = t->toBasetype();
    return t->ty == Tsarray ||
           (t->ty == Tstruct && t->size() > 16 && !isHFVA(t));
  }

public:
  AArch64TargetABI() : isDarwin(global.params.targetTriple->isOSDarwin()) {}

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    return passIndirectlyByValue(rt);
  }

  bool passByVal(TypeFunction *, Type *) override { return false; }

  void rewriteFunctionType(IrFuncTy &fty) override {
    Type *rt = fty.ret->type->toBasetype();
    if (!fty.ret->byref && rt->ty != Tvoid) {
      rewriteArgument(fty, *fty.ret, true);
    }

    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg, bool isReturnVal) {
    const auto t = arg.type->toBasetype();

    if (t->ty != Tstruct && t->ty != Tsarray)
      return;

    if (!isReturnVal && isAAPCS64VaList(t)) {
      // compiler magic: pass va_list args implicitly by reference
      arg.byref = true;
      arg.ltype = arg.ltype->getPointerTo();
    } else if (!isReturnVal && passIndirectlyByValue(t)) {
      byvalRewrite.applyTo(arg);
    }
    // Rewrite HFAs only because union HFAs are turned into IR types that are
    // non-HFA and messes up register selection
    else if (t->ty == Tstruct && isHFVA(t, &arg.ltype)) {
      hfvaToArray.applyTo(arg, arg.ltype);
    } else {
      if (isReturnVal) {
        integerRewrite.applyTo(arg);
      } else {
        compositeToArray64.applyTo(arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    rewriteArgument(fty, arg, false);
  }

  Type *vaListType() override {
    if (isDarwin)
      return TargetABI::vaListType(); // char*

    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a better
    // solution is found there, this should be adapted).
    return createTypeIdentifier(Loc(), Identifier::idPool("__va_list"));
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    return "objc_msgSend";
  }
};

// The public getter for abi.cpp
TargetABI *getAArch64TargetABI() { return new AArch64TargetABI(); }
