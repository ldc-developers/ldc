//===-- gen/abi-loongarch64.cpp - LoongArch64 ABI description -----------*- C++
//-*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ABI spec:
// https://github.com/loongson/la-abi-specs/blob/release/lapcs.adoc
//
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

using namespace dmd;

namespace {
struct Integer2Rewrite : BaseBitcastABIRewrite {
  LLType *type(Type *t) override {
    return LLStructType::get(gIR->context(),
                             {DtoType(Type::tint64), DtoType(Type::tint64)});
  }
};

struct FlattenedFields {
  Type *fields[2];
  int length = 0; // use -1 to represent "no need to rewrite" condition
};

FlattenedFields visitStructFields(Type *ty, unsigned baseOffset) {
  // recursively visit a POD struct to flatten it
  FlattenedFields result;
  if (auto ts = ty->isTypeStruct()) {
    for (auto fi : ts->sym->fields) {
      auto sub =
          visitStructFields(fi->type->toBasetype(), baseOffset + fi->offset);
      if (sub.length == -1 || result.length + sub.length > 2) {
        result.length = -1;
        return result;
      }
      for (unsigned i = 0; i < (unsigned)sub.length; ++i) {
        result.fields[result.length++] = sub.fields[i];
      }
    }
    return result;
  }
  switch (ty->ty) {
  case TY::Tcomplex32: // treat it as {float32, float32}
    result.fields[0] = pointerTo(Type::tfloat32);
    result.fields[1] = pointerTo(Type::tfloat32);
    result.length = 2;
    break;
  case TY::Tcomplex64: // treat it as {float64, float64}
    result.fields[0] = pointerTo(Type::tfloat64);
    result.fields[1] = pointerTo(Type::tfloat64);
    result.length = 2;
    break;
  default:
    if (ty->size() > 8) {
      // field larger than GRLEN and FRLEN
      result.length = -1;
      break;
    }
    result.fields[0] = ty;
    result.length = 1;
    break;
  }
  return result;
}

struct HardfloatRewrite : BaseBitcastABIRewrite {
  LLType *type(Type *ty, const FlattenedFields &flat) {
    if (flat.length == 1) {
      return LLStructType::get(gIR->context(), {DtoType(flat.fields[0])},
                               false);
    }
    assert(flat.length == 2);
    LLType *t[2];
    for (unsigned i = 0; i < 2; ++i) {
      t[i] =
          flat.fields[i]->isfloating()
              ? DtoType(flat.fields[i])
              : LLIntegerType::get(gIR->context(), flat.fields[i]->size() * 8);
    }
    return LLStructType::get(gIR->context(), {t[0], t[1]}, false);
  }
  LLType *type(Type *ty) override {
    return type(ty, visitStructFields(ty->toBasetype(), 0));
  }
};
} // anonymous namespace

struct LoongArch64TargetABI : TargetABI {
private:
  HardfloatRewrite hardfloatRewrite;
  IndirectByvalRewrite indirectByvalRewrite{};
  Integer2Rewrite integer2Rewrite;
  IntegerRewrite integerRewrite;

  bool requireHardfloatRewrite(Type *ty) {
    if (!isPOD(ty) || !ty->toBasetype()->isTypeStruct())
      return false;
    auto result = visitStructFields(ty->toBasetype(), 0);
    if (result.length <= 0)
      return false;
    if (result.length == 1)
      return result.fields[0]->isfloating();
    return result.fields[0]->isfloating() || result.fields[1]->isfloating();
  }

public:
  auto returnInArg(TypeFunction *tf, bool) -> bool override {
    if (tf->isref()) {
      return false;
    }
    Type *rt = tf->next->toBasetype();
    if (!isPOD(rt)) {
      return true;
    }
    // pass by reference when > 2*GRLEN
    return rt->size() > 16;
  }

  auto passByVal(TypeFunction *, Type *t) -> bool override {
    if (!isPOD(t)) {
      return false;
    }
    return t->size() > 16;
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!skipReturnValueRewrite(fty)) {
      if (requireHardfloatRewrite(fty.ret->type)) {
        // rewrite here because we should not apply this to variadic arguments
        hardfloatRewrite.applyTo(*fty.ret);
      } else {
        rewriteArgument(fty, *fty.ret);
      }
    }

    for (auto arg : fty.args) {
      if (!arg->byref) {
        if (requireHardfloatRewrite(arg->type)) {
          // rewrite here because we should not apply this to variadic arguments
          hardfloatRewrite.applyTo(*arg);
        } else {
          rewriteArgument(fty, *arg);
        }
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    if (!isPOD(arg.type)) {
      // non-PODs should be passed in memory
      indirectByvalRewrite.applyTo(arg);
      return;
    }

    Type *ty = arg.type->toBasetype();
    if (ty->isintegral() && (ty->ty == TY::Tint32 || ty->ty == TY::Tuns32 ||
                             ty->ty == TY::Tdchar)) {
      // In the LP64D ABI, both int32 and unsigned int32 are stored in
      // general-purpose registers as proper sign extensions of their
      // 32-bit values. So, the native ABI function's int32 arguments and
      // return values should have the `signext` attribute.
      // C example: https://godbolt.org/z/vcjErxj76
      arg.attrs.addAttribute(LLAttribute::SExt);
    } else if (isAggregate(ty) && ty->size() && ty->size() <= 16) {
      if (ty->size() > 8 && DtoAlignment(ty) < 16) {
        // pass the aggregate as {int64, int64} to avoid wrong alignment
        integer2Rewrite.applyToIfNotObsolete(arg);
      } else {
        integerRewrite.applyToIfNotObsolete(arg);
      }
    }
  }
};

// The public getter for abi.cpp
TargetABI *getLoongArch64TargetABI() { return new LoongArch64TargetABI(); }
