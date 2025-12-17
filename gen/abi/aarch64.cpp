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

#include "dmd/identifier.h"
#include "dmd/nspace.h"
#include "gen/abi/abi.h"
#include "gen/abi/generic.h"

using namespace dmd;

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
  IndirectByvalRewrite indirectByvalRewrite;
  ArgTypesRewrite argTypesRewrite;

  bool hasAAPCS64VaList() {
    return !isDarwin() &&
           !global.params.targetTriple->isWindowsMSVCEnvironment();
  }

  bool isAAPCS64VaList(Type *t) {
    if (!hasAAPCS64VaList())
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
  AArch64TargetABI() {}

  llvm::UWTableKind defaultUnwindTableKind() override {
    return isDarwin() ? llvm::UWTableKind::Sync : llvm::UWTableKind::Async;
  }

  bool returnInArg(TypeFunction *tf, bool) override {
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
    TargetABI::rewriteFunctionType(fty);

    // Darwin: remove empty POD structs
    // https://developer.apple.com/library/archive/documentation/Xcode/Conceptual/iPhoneOSABIReference/Articles/ARM64FunctionCallingConventions.html#//apple_ref/doc/uid/TP40013702-SW1
    if (isDarwin()) {
      size_t i = 0;
      while (i < fty.args.size()) {
        auto arg = fty.args[i];
        if (!arg->byref) {
          if (auto ts = arg->type->toBasetype()->isTypeStruct()) {
            if (ts->sym->fields.empty() && dmd::isPOD(ts->sym)) {
              fty.args.erase(fty.args.begin() + i);
              continue;
            }
          }
        }
        ++i;
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *t = arg.type->toBasetype();

    if (!isAggregate(t))
      return;

    const bool isReturnVal = &arg == fty.ret;

    // compiler magic: pass va_list args implicitly by reference
    if (!isReturnVal && isAAPCS64VaList(t)) {
      arg.byref = true;
      arg.ltype = LLPointerType::get(getGlobalContext(), 0);
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
    if (!hasAAPCS64VaList())
      return TargetABI::vaListType(); // char*

    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a
    // better solution is found there, this should be adapted).
    return TypeIdentifier::create(Loc(), Identifier::idPool("__va_list"));
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty, bool directcall) override {
    assert(isDarwin());
    
    // see objc/message.h for objc_msgSend selection rules
    return directcall ? "objc_msgSendSuper" : "objc_msgSend";
  }
};

// The public getter for abi.cpp
TargetABI *getAArch64TargetABI() { return new AArch64TargetABI(); }
