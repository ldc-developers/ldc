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
//
//===----------------------------------------------------------------------===//

#include "ldcbindings.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-aarch64.h"

/**
 * The AACPS64 uses a special native va_list type:
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
  HFAToArray hfaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    return passByVal(tf, rt);
  }

  bool isVaList(Type *t) {
    return t->ty == Tstruct && strcmp(t->toPrettyChars(true),
                                      "ldc.internal.vararg.std.__va_list") == 0;
  }

  bool passByVal(TypeFunction *, Type *t) override {
    t = t->toBasetype();
    return t->ty == Tsarray || (t->ty == Tstruct && t->size() > 16 &&
                                !isHFA((TypeStruct *)t) && !isVaList(t));
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    Type *retTy = fty.ret->type->toBasetype();
    if (!fty.ret->byref && retTy->ty == Tstruct) {
      // Rewrite HFAs only because union HFAs are turned into IR types that are
      // non-HFA and messes up register selection
      if (isHFA((TypeStruct *)retTy, &fty.ret->ltype)) {
        hfaToArray.applyTo(*fty.ret, fty.ret->ltype);
      } else {
        integerRewrite.applyTo(*fty.ret);
      }
    }

    for (auto arg : fty.args) {
      if (!arg->byref)
        rewriteArgument(fty, *arg);
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    // FIXME
    Type *ty = arg.type->toBasetype();

    if (ty->ty == Tstruct || ty->ty == Tsarray) {
      if (isVaList(ty)) {
        // compiler magic: pass va_list args implicitly by reference
        arg.byref = true;
        arg.ltype = arg.ltype->getPointerTo();
      }
      // Rewrite HFAs only because union HFAs are turned into IR types that are
      // non-HFA and messes up register selection
      else if (ty->ty == Tstruct && isHFA((TypeStruct *)ty, &arg.ltype)) {
        hfaToArray.applyTo(arg, arg.ltype);
      } else {
        compositeToArray64.applyTo(arg);
      }
    }
  }

  Type *vaListType() override {
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
