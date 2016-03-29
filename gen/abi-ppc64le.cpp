//===-- abi-ppc64.cpp -----------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit little-endian PowerPC targets.
//
// The PowerOpen 64bit ELF v2 ABI can be found here:
// https://members.openpowerfoundation.org/document/dl/576
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-ppc64le.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct PPC64LETargetABI : TargetABI {
  HFAToArray hfaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  explicit PPC64LETargetABI() : hfaToArray(8) {}

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    // FIXME: The return value of this function translates
    // to RETstack or RETregs in function retStyle(), which
    // directly influences if NRVO is possible or not
    // (false -> RETregs -> nrvo_can = false). Depending on
    // NRVO, the postblit constructor is called or not.
    // Thus using the rules of the C ABI here (as mandated by
    // the D specification) leads to crashes.
    if (tf->linkage == LINKd)
      return rt->ty == Tsarray || rt->ty == Tstruct;

    return rt->ty == Tsarray || (rt->ty == Tstruct && rt->size() > 16 &&
                                 !isHFA((TypeStruct *)rt, nullptr, 8));
  }

  bool passByVal(Type *t) override {
    t = t->toBasetype();
    return t->ty == Tsarray || (t->ty == Tstruct && t->size() > 16 &&
                                !isHFA((TypeStruct *)t, nullptr, 8));
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    // RETURN VALUE
    Type *retTy = fty.ret->type->toBasetype();
    if (!fty.ret->byref) {
      if (retTy->ty == Tstruct || retTy->ty == Tsarray) {
        if (retTy->ty == Tstruct &&
            isHFA((TypeStruct *)retTy, &fty.ret->ltype, 8)) {
          fty.ret->rewrite = &hfaToArray;
          fty.ret->ltype = hfaToArray.type(fty.ret->type, fty.ret->ltype);
        } else if (canRewriteAsInt(retTy, true)) {
          fty.ret->rewrite = &integerRewrite;
          fty.ret->ltype = integerRewrite.type(fty.ret->type, fty.ret->ltype);
        } else {
          fty.ret->rewrite = &compositeToArray64;
          fty.ret->ltype =
              compositeToArray64.type(fty.ret->type, fty.ret->ltype);
        }
      } else if (retTy->isintegral())
        fty.ret->attrs.add(retTy->isunsigned() ? LLAttribute::ZExt
                                               : LLAttribute::SExt);
    }

    // EXPLICIT PARAMETERS
    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *ty = arg.type->toBasetype();
    if (ty->ty == Tstruct || ty->ty == Tsarray) {
      if (ty->ty == Tstruct && isHFA((TypeStruct *)ty, &arg.ltype, 8)) {
        arg.rewrite = &hfaToArray;
        arg.ltype = hfaToArray.type(arg.type, arg.ltype);
      } else if (canRewriteAsInt(ty, true)) {
        arg.rewrite = &integerRewrite;
        arg.ltype = integerRewrite.type(arg.type, arg.ltype);
      } else {
        arg.rewrite = &compositeToArray64;
        arg.ltype = compositeToArray64.type(arg.type, arg.ltype);
      }
    } else if (ty->isintegral())
      arg.attrs.add(ty->isunsigned() ? LLAttribute::ZExt : LLAttribute::SExt);
  }
};

// The public getter for abi.cpp
TargetABI *getPPC64LETargetABI() { return new PPC64LETargetABI(); }
