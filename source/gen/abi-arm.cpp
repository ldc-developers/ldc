//===-- abi-arm.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

/*
  ARM ABI based on AAPCS (Procedure Call Standard for the ARM Architecture)

  http://infocenter.arm.com/help/topic/com.arm.doc.ihi0042f/IHI0042F_aapcs.pdf
*/

#include "ldcbindings.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-arm.h"
#include "llvm/Target/TargetMachine.h"

struct ArmTargetABI : TargetABI {
  HFAToArray hfaToArray;
  CompositeToArray32 compositeToArray32;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  bool returnInArg(TypeFunction *tf) override {
    // AAPCS 5.4 wants composites > 4-bytes returned by arg except for
    // Homogeneous Aggregates of up-to 4 float types (6.1.2.1) - an HFA.
    // TODO: see if Tsarray should be candidate for HFA.
    if (tf->isref)
      return false;
    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    return rt->ty == Tsarray ||
           (rt->ty == Tstruct && rt->size() > 4 &&
             (gTargetMachine->Options.FloatABIType == llvm::FloatABI::Soft ||
             !isHFA((TypeStruct *)rt)));
  }

  bool passByVal(Type *t) override {
    // AAPCS does not use an indirect arg to pass aggregates, however
    // clang uses byval for types > 64-bytes, then llvm backend
    // converts back to non-byval.  Without this special handling the
    // optimzer generates bad code (e.g. std.random unittest crash).
    t = t->toBasetype();
    return ((t->ty == Tsarray || t->ty == Tstruct) && t->size() > 64);

    // Note: byval can have a codegen problem with -O1 and higher.
    // What happens is that load instructions are being incorrectly
    // reordered before stores.  It is a problem in the LLVM backend.
    // The outcome is a program with incorrect results or crashes.
    // It happens in the "top-down list latency scheduler" pass
    //
    //   https://forum.dlang.org/post/m2r3u5ac0c.fsf@comcast.net
    //
    // Revist and determine if the byval problem is only for small
    // structs, say 16-bytes or less, that can entirely fit in
    // registers.

    // Note: the codegen is horrible for Tsarrays passed this way -
    // does a copy without a loop for huge arrays.  Could be better if
    // byval was always used for sarrays, and maybe can if above
    // problem is better understood.
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    Type *retTy = fty.ret->type->toBasetype();
    if (!fty.ret->byref && retTy->ty == Tstruct) {
      // Rewrite HFAs only because union HFAs are turned into IR types that are
      // non-HFA and messes up register selection
      if (isHFA((TypeStruct *)retTy, &fty.ret->ltype)) {
        fty.ret->rewrite = &hfaToArray;
      } else {
        fty.ret->rewrite = &integerRewrite;
        fty.ret->ltype = integerRewrite.type(fty.ret->type);
      }
    }

    for (auto arg : fty.args) {
      if (!arg->byref)
        rewriteArgument(fty, *arg);
    }

    // extern(D): reverse parameter order for non variadics, for DMD-compliance
    if (tf->linkage == LINKd && tf->varargs != 1 && fty.args.size() > 1) {
      fty.reverseParams = true;
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    // structs and arrays need rewrite as i32 arrays.  This keeps data layout
    // unchanged when passed in registers r0-r3 and is necessary to match C ABI
    // for struct passing.  Without out this rewrite, each field or array
    // element is passed in own register.  For example: char[4] now all fits in
    // r0, where before it consumed r0-r3.
    Type *ty = arg.type->toBasetype();

    // TODO: want to also rewrite Tsarray as i32 arrays, but sometimes
    // llvm selects an aligned ldrd instruction even though the ptr is
    // unaligned (e.g. walking through members of array char[5][]).
    // if (ty->ty == Tstruct || ty->ty == Tsarray)
    if (ty->ty == Tstruct) {
      // Rewrite HFAs only because union HFAs are turned into IR types that are
      // non-HFA and messes up register selection
      if (isHFA((TypeStruct *)ty, &arg.ltype)) {
        arg.rewrite = &hfaToArray;
      } else if (DtoAlignment(ty) <= 4) {
        arg.rewrite = &compositeToArray32;
        arg.ltype = compositeToArray32.type(arg.type);
      } else {
        arg.rewrite = &compositeToArray64;
        arg.ltype = compositeToArray64.type(arg.type);
      }
    }
  }

  Type *vaListType() override {
    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a better
    // solution is found there, this should be adapted).
    static const llvm::StringRef ident = "__va_list";
    return (createTypeIdentifier(
        Loc(), Identifier::idPool(ident.data(), ident.size())));
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    if (fty.arg_sret) {
      return "objc_msgSend_stret";
    }
    return "objc_msgSend";
  }
};

TargetABI *getArmTargetABI() { return new ArmTargetABI; }
