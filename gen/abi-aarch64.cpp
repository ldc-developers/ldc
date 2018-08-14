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

#include "gen/abi-aarch64.h"
#include "gen/abi-generic.h"
#include "gen/abi.h"

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
private:
  ExplicitByvalRewrite byvalRewrite;
  HFAToArray hfaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  bool isVaList(Type *t) {
    return t->ty == Tstruct && strcmp(t->toPrettyChars(true),
                                      "ldc.internal.vararg.std.__va_list") == 0;
  }

  bool passByValExplicit(Type *t) {
    t = t->toBasetype();
    return t->ty == Tsarray ||
           (t->ty == Tstruct && t->size() > 16 && !isHFA((TypeStruct *)t));
  }

public:
  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    return passByValExplicit(rt);
  }

  bool passByVal(Type *t) override { return false; }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    Type *retTy = fty.ret->type->toBasetype();
    if (!fty.ret->byref && retTy->ty == Tstruct) {
      rewriteArgument(fty, *fty.ret, true);
    }

    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }

    // extern(D): reverse parameter order for non variadics, for DMD-compliance
    if (tf->linkage == LINKd && tf->varargs != 1 && fty.args.size() > 1) {
      fty.reverseParams = true;
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg, bool isReturnVal) {
    const auto t = arg.type->toBasetype();

    if (t->ty != Tstruct && t->ty != Tsarray)
      return;

    if (!isReturnVal && isVaList(t)) {
      // compiler magic: pass va_list args implicitly by reference
      arg.byref = true;
      arg.ltype = arg.ltype->getPointerTo();
    } else if (!isReturnVal && passByValExplicit(t)) {
      arg.rewrite = &byvalRewrite;
      arg.attrs.clear()
          .add(LLAttribute::NoAlias)
          .add(LLAttribute::NoCapture)
          .addAlignment(byvalRewrite.alignment(arg.type));
    } else if (t->ty == Tstruct &&
               isHFA(static_cast<TypeStruct *>(t), &arg.ltype)) {
      arg.rewrite = &hfaToArray;
    } else {
      arg.rewrite = &compositeToArray64;
    }

    if (arg.rewrite) {
      const auto originalLType = arg.ltype;
      arg.ltype = arg.rewrite->type(arg.type, arg.ltype);

      IF_LOG {
        Logger::println("Rewriting argument type %s", t->toChars());
        LOG_SCOPE;
        Logger::cout() << *originalLType << " => " << *arg.ltype << '\n';
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    rewriteArgument(fty, arg, false);
  }

  void vaCopy(LLValue *pDest, LLValue *src) override {
    // `src` is a __va_list* pointer (because it's a struct).
    DtoMemCpy(pDest, src);
  }

  Type *vaListType() override {
    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a better
    // solution is found there, this should be adapted).
    return (new TypeIdentifier(Loc(), Identifier::idPool("__va_list")));
  }
};

// The public getter for abi.cpp
TargetABI *getAArch64TargetABI() { return new AArch64TargetABI(); }
