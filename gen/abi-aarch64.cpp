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
 * Based on https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst.
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
  ImplicitByvalRewrite byvalRewrite;
  HFVAToArray hfvaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  RegCount &getRegCount(IrFuncTy &fty) {
    return reinterpret_cast<RegCount &>(fty.tag);
  }

  bool isAAPCS64VaList(Type *t) {
    return !isDarwin && t->ty == Tstruct &&
           strcmp(t->toPrettyChars(true),
                  "ldc.internal.vararg.std.__va_list") == 0;
  }

  bool passIndirectlyByValue(Type *t) {
    t = t->toBasetype();
    return (t->ty == Tstruct || t->ty == Tsarray) &&
           (!isPOD(t) || (t->size() > 16 && !isHFVA(t)));
  }

public:
  AArch64TargetABI() : isDarwin(global.params.targetTriple->isOSDarwin()) {}

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();
    return passIndirectlyByValue(rt);
  }

  bool passByVal(TypeFunction *, Type *) override { return false; }

  void rewriteFunctionType(IrFuncTy &fty) override {
    RegCount &regCount = getRegCount(fty);
    regCount = RegCount(8, 8); // initialize

    // return value
    if (!fty.ret->byref && fty.ret->type->toBasetype()->ty != Tvoid) {
      RegCount dummy = regCount;
      rewriteArgument(fty, *fty.ret, dummy, /*isReturnVal=*/true);
    }

    // implicit parameters taking up GP registers
    /* the sret pointer is passed in a dedicated register (x8)
    if (fty.arg_sret)
      regCount.gp_regs--;
    */
    if (fty.arg_this || fty.arg_nest)
      regCount.gp_regs--;
    if (fty.arg_objcSelector)
      regCount.gp_regs--;
    if (fty.arg_arguments)
      regCount.gp_regs -= 2; // slice

    int begin = 0, end = fty.args.size(), step = 1;
    if (fty.reverseParams) {
      begin = end - 1;
      end = -1;
      step = -1;
    }
    for (int i = begin; i != end; i += step) {
      IrFuncTyArg &arg = *fty.args[i];

      if (arg.byref) {
        if (regCount.gp_regs > 0 && !arg.isByVal())
          regCount.gp_regs--;

        continue;
      }

      rewriteArgument(fty, arg, regCount);
    }

    // regCount (fty.tag) is now in the state after all implicit & formal args,
    // ready to serve as initial state for each vararg call site, see below
  }

  void rewriteVarargs(IrFuncTy &fty,
                      std::vector<IrFuncTyArg *> &args) override {
    // use a dedicated RegCount copy for each call site and initialize it with
    // fty.tag
    RegCount regCount = getRegCount(fty);

    for (auto arg : args) {
      if (!arg->byref) // don't rewrite ByVal arguments
        rewriteArgument(fty, *arg, regCount);
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    llvm_unreachable("Please use the other overload explicitly.");
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg, RegCount &regCount,
                       bool isReturnVal = false) {
    LLType *originalLType = arg.ltype;
    Type *t = arg.type->toBasetype();

    // compiler magic: pass va_list args implicitly by reference
    if (!isReturnVal && isAAPCS64VaList(t)) {
      arg.byref = true;
      arg.ltype = arg.ltype->getPointerTo();
      if (regCount.gp_regs > 0)
        regCount.gp_regs--;

      return;
    }

    // non-PODs and bigger aggregates are passed as pointer to hidden copy
    if (passIndirectlyByValue(t)) {
      indirectByvalRewrite.applyTo(arg);
      if (regCount.gp_regs > 0)
        regCount.gp_regs--;

      return;
    }

    LLType *hfvaType = nullptr;
    if (isHFVA(t, &hfvaType)) {
      // pass in SIMD registers (if enough are available for the whole aggregate)
      hfvaToArray.applyToIfNotObsolete(arg, hfvaType);
    } else if (t->ty == Tstruct || t->ty == Tsarray) {
      // pass remaining structs in 1 or 2 GP registers (if enough are available)
      if (canRewriteAsInt(t)) {
        integerRewrite.applyToIfNotObsolete(arg);
      } else {
        compositeToArray64.applyToIfNotObsolete(arg);
      }
      /* if 16-bytes aligned, the 1st GP register must be an even one
      if ((regCount.gp_regs & 1) && DtoAlignment(t) >= 16)
        regCount.gp_regs--;
      */
    }

    if (regCount.trySubtract(arg) == RegCount::ArgumentWouldFitInPartially) {
      // pass the LL aggregate with byval attribute to prevent LLVM from passing
      // it partially in registers, partially in memory
      assert(originalLType->isAggregateType());
      IF_LOG Logger::cout()
          << "Passing byval to prevent register/memory mix: "
          << arg.type->toChars() << " (" << *originalLType << ")\n";
      byvalRewrite.applyTo(arg);
    }
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
