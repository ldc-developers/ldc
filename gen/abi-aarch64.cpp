//===-- abi-aarch64.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The AArch64 Procedure Call Standard (AAPCS64) can be found here:
// https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst
//
//===----------------------------------------------------------------------===//

#include "gen/abi-aarch64.h"

#include "dmd/identifier.h"
#include "dmd/nspace.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"

/**
 * AAPCS64 uses a special native va_list type, a struct aliased as
 * object.__va_list in druntime. Apple diverges and uses a simple char*
 * pointer.
 * __va_list arguments are never passed by value, only by reference (even though
 * the mangled function name indicates otherwise!). This requires a little bit
 * of compiler magic in the following implementations.
 */
struct AArch64TargetABI : TargetABI {
private:
  const bool isDarwin;
  IndirectByvalRewrite indirectByvalRewrite;
  ArgTypesRewrite argTypesRewrite;

  bool isAAPCS64VaList(Type *t) {
    if (isDarwin)
      return false;

    // look for a __va_list struct in a `std` C++ namespace
    if (auto ts = t->isTypeStruct()) {
      auto sd = ts->sym;
      if (strcmp(sd->ident->toChars(), "__va_list") == 0) {
        if (auto ns = sd->parent->isNspace()) {
          return strcmp(ns->toChars(), "std") == 0;
        }
      }
    }

    return false;
  }

public:
  AArch64TargetABI() : isDarwin(global.params.targetTriple->isOSDarwin()) {}

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref()) {
      return false;
    }

    Type *rt = tf->next->toBasetype();
    if (rt->ty == TY::Tstruct || rt->ty == TY::Tsarray) {
      auto argTypes = getArgTypes(rt);
      return !argTypes // FIXME: get rid of sret workaround for 0-sized return
                       //        values (static arrays with 0 elements)
             || argTypes->arguments->empty();
    }

    return false;
  }

  // Prefer a ref if the POD cannot be passed in registers, i.e., if
  // IndirectByvalRewrite would be applied.
  bool preferPassByRef(Type *t) override {
    t = t->toBasetype();

    if (!(t->ty == TY::Tstruct || t->ty == TY::Tsarray))
      return false;

    auto argTypes = getArgTypes(t);
    return argTypes // not 0-sized
        && argTypes->arguments->empty(); // cannot be passed in registers
  }

  bool passByVal(TypeFunction *, Type *) override { return false; }

  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!skipReturnValueRewrite(fty)) {
      rewriteArgument(fty, *fty.ret, /*isReturnVal=*/true);
    }

    for (auto arg : fty.args) {
      if (!arg->byref)
        rewriteArgument(fty, *arg, /*isReturnVal=*/false);
    }

    // remove 0-sized args (static arrays with 0 elements) and, for Darwin,
    // empty POD structs too
    size_t i = 0;
    while (i < fty.args.size()) {
      auto arg = fty.args[i];
      if (!arg->byref) {
        auto tb = arg->type->toBasetype();

        if (tb->size() == 0) {
          fty.args.erase(fty.args.begin() + i);
          continue;
        }

        // https://developer.apple.com/library/archive/documentation/Xcode/Conceptual/iPhoneOSABIReference/Articles/ARM64FunctionCallingConventions.html#//apple_ref/doc/uid/TP40013702-SW1
        if (isDarwin) {
          if (auto ts = tb->isTypeStruct()) {
            if (ts->sym->fields.empty() && ts->sym->isPOD()) {
              fty.args.erase(fty.args.begin() + i);
              continue;
            }
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
      // in most cases, a LL array of either floats/vectors (HFVAs) or i64
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
    return TypeIdentifier::create(Loc(), Identifier::idPool("__va_list"));
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    return "objc_msgSend";
  }
};

// The public getter for abi.cpp
TargetABI *getAArch64TargetABI() { return new AArch64TargetABI(); }
