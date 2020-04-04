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
 * Based on
 * https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst.
 *
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
  IndirectByvalRewrite indirectByvalRewrite;
  ArgTypesRewrite argTypesRewrite;

  bool isAAPCS64VaList(Type *t) {
    return !isDarwin && t->ty == Tstruct &&
           strcmp(t->toPrettyChars(true),
                  "ldc.internal.vararg.std.__va_list") == 0;
  }

public:
  AArch64TargetABI() : isDarwin(global.params.targetTriple->isOSDarwin()) {}

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref)
      return false;

    Type *rt = tf->next->toBasetype();
    if (rt->ty == Tstruct || rt->ty == Tsarray) {
      auto argTypes = getArgTypes(rt);
      return !argTypes // TODO: no sret workaround for 0-sized return values
                       // (static arrays with 0 elements)
             || argTypes->arguments->empty();
    }

    return false;
  }

  bool passByVal(TypeFunction *, Type *) override { return false; }

  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!fty.ret->byref && fty.ret->type->toBasetype()->ty != Tvoid) {
      rewriteArgument(fty, *fty.ret, /*isReturnVal=*/true);
    }

    for (auto arg : fty.args) {
      if (!arg->byref)
        rewriteArgument(fty, *arg, /*isReturnVal=*/false);
    }

    // remove 0-sized args (static arrays with 0 elements) and, for Darwin,
    // empty structs too
    size_t i = 0;
    while (i < fty.args.size()) {
      auto tb = fty.args[i]->type->toBasetype();

      if (tb->size() == 0) {
        fty.args.erase(fty.args.begin() + i);
        continue;
      }

      // https://developer.apple.com/library/archive/documentation/Xcode/Conceptual/iPhoneOSABIReference/Articles/ARM64FunctionCallingConventions.html#//apple_ref/doc/uid/TP40013702-SW1
      if (isDarwin) {
        if (auto ts = tb->isTypeStruct()) {
          if (ts->sym->fields.empty()) {
            fty.args.erase(fty.args.begin() + i);
            continue;
          }
        }
      }

      ++i;
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    return rewriteArgument(fty, arg, /*isReturnVal=*/false);
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg, bool isReturnVal) {
    Type *t = arg.type->toBasetype();

    if (!isAggregate(t))
      return;

    // compiler magic: pass va_list args implicitly by reference
    if (!isReturnVal && isAAPCS64VaList(t)) {
      arg.byref = true;
      arg.ltype = arg.ltype->getPointerTo();
      return;
    }

    auto argTypes = getArgTypes(t);
    if (!argTypes)
      return; // don't rewrite 0-sized types

    if (argTypes->arguments->empty()) {
      // non-PODs and larger non-HFVA aggregates are passed as pointer to
      // hidden copy
      indirectByvalRewrite.applyTo(arg);
      return;
    }

    // LLVM seems to take care of the rest when rewriting as follows, close to
    // what clang emits:

    auto rewrittenType = getRewrittenArgType(t, argTypes);
    if (!rewrittenType)
      return;

    if (rewrittenType->isIntegerTy()) {
      argTypesRewrite.applyToIfNotObsolete(arg, rewrittenType);
    } else {
      argTypesRewrite.applyTo(arg, rewrittenType);
    }
  }

  Type *vaListType() override {
    if (isDarwin)
      return TargetABI::vaListType(); // char*

    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a
    // better solution is found there, this should be adapted).
    return createTypeIdentifier(Loc(), Identifier::idPool("__va_list"));
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    return "objc_msgSend";
  }
};

// The public getter for abi.cpp
TargetABI *getAArch64TargetABI() { return new AArch64TargetABI(); }
