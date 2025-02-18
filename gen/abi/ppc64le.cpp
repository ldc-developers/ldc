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
// https://files.openpower.foundation/s/cfA2oFPXbbZwEBK/download/64biteflv2abi-v1.5.pdf
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

using namespace dmd;

struct PPC64LETargetABI : TargetABI {
  HFVAToArray hfvaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  explicit PPC64LETargetABI() : hfvaToArray(8) {}

  bool passByVal(TypeFunction *, Type *t) override {
    t = t->toBasetype();
    return isPOD(t) &&
           (t->ty == TY::Tsarray || (t->ty == TY::Tstruct && size(t) > 16 &&
                                     !isHFVA(t, hfvaToArray.maxElements)));
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    TargetABI::rewriteArgument(fty, arg);
    if (arg.rewrite)
      return;

    Type *ty = arg.type->toBasetype();
    if (ty->ty == TY::Tstruct || ty->ty == TY::Tsarray) {
      if (ty->ty == TY::Tstruct &&
          isHFVA(ty, hfvaToArray.maxElements, &arg.ltype)) {
        hfvaToArray.applyTo(arg, arg.ltype);
      } else if (canRewriteAsInt(ty, true)) {
        integerRewrite.applyTo(arg);
      } else {
        compositeToArray64.applyTo(arg);
      }
    } else if (ty->isintegral() && !ty->isTypeVector()) {
      arg.attrs.addAttribute(ty->isunsigned() ? LLAttribute::ZExt
                                              : LLAttribute::SExt);
    }
  }
};

// The public getter for abi.cpp
TargetABI *getPPC64LETargetABI() { return new PPC64LETargetABI(); }
