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

#include <llvm/IR/IntrinsicsPowerPC.h>

#include "driver/cl_options.h"
#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

using namespace dmd;

struct LongDoubleRewrite : ABIRewrite {
  inline bool shouldRewrite(Type *ty) {
    const auto baseTy = ty->toBasetype()->ty;
    return baseTy == TY::Tfloat80 || baseTy == TY::Tcomplex80 ||
           baseTy == TY::Timaginary80;
  }

  LLValue *put(DValue *dv, bool, bool) override {
    if (shouldRewrite(dv->type)) {
      auto *dconst = llvm::dyn_cast<llvm::ConstantFP>(DtoRVal(dv));
      if (dconst) {
        // try to CTFE the conversion
        // (ppc_convert_f128_to_ppcf128 intrinsics do not perform the conversion
        // during the compile time)
        bool ignored;
        auto apfloat = dconst->getValue();
        apfloat.convert(llvm::APFloat::PPCDoubleDouble(),
                        llvm::APFloat::rmNearestTiesToEven, &ignored);
        return llvm::ConstantFP::get(gIR->context(), apfloat);
      }
      const auto convertFunc = llvm::Intrinsic::getDeclaration(
          &gIR->module, llvm::Intrinsic::ppc_convert_f128_to_ppcf128);
      const auto ret = gIR->funcGen().callOrInvoke(
          convertFunc, convertFunc->getFunctionType(), {DtoRVal(dv)});
      return ret;
    }
    return DtoRVal(dv);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    // inverse operation of method "put"
    auto dstType = DtoType(dty);
    if (shouldRewrite(dty) && dstType->isFP128Ty()) {
      const auto convertFunc = llvm::Intrinsic::getDeclaration(
          &gIR->module, llvm::Intrinsic::ppc_convert_ppcf128_to_f128);
      const auto retType = LLType::getFP128Ty(gIR->context());
      const auto buffer = DtoRawAlloca(retType, 16);
      const auto ret = gIR->funcGen().callOrInvoke(
          convertFunc, convertFunc->getFunctionType(), {v});
      DtoStore(ret, buffer);
      return buffer;
    }
    // dual-ABI situation: if the destination type is already correct, we just store it
    const auto buffer = DtoRawAlloca(dstType, 16);
    DtoStore(v, buffer);
    return buffer;
  }

  LLType *type(Type *ty) override {
    return LLType::getPPC_FP128Ty(gIR->context());
  }
};

struct PPC64LETargetABI : TargetABI {
  HFVAToArray hfvaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;
  LongDoubleRewrite longDoubleRewrite;
  bool useIEEE128;

  explicit PPC64LETargetABI()
      : hfvaToArray(8), useIEEE128(opts::mABI == "ieeelongdouble") {}

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
    } else if (ty->isintegral()) {
      arg.attrs.addAttribute(ty->isunsigned() ? LLAttribute::ZExt
                                              : LLAttribute::SExt);
    } else if (!useIEEE128 && longDoubleRewrite.shouldRewrite(arg.type)) {
      longDoubleRewrite.applyTo(arg);
    }
  }
};

// The public getter for abi.cpp
TargetABI *getPPC64LETargetABI() { return new PPC64LETargetABI(); }
