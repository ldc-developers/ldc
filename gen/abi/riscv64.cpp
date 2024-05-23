//===-- gen/abi-riscv64.cpp - RISCV64 ABI description -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ABI spec:
// https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-cc.adoc
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
  struct FlattenedField {
    Type *ty = nullptr;
    unsigned offset = 0;
  };
  FlattenedField fields[2];
  int length = 0; // use -1 to represent "no need to rewrite" condition
};

FlattenedFields visitStructFields(Type *ty, unsigned baseOffset) {
  // recursively visit a POD struct to flatten it
  // FIXME: may cause low performance
  // dmd may cache argtypes in some other architectures as a TypeTuple, but we
  // need to additionally store field offsets to realign later
  FlattenedFields result;
  if (auto ts = ty->toBasetype()->isTypeStruct()) {
    for (auto fi : ts->sym->fields) {
      auto sub = visitStructFields(fi->type, baseOffset + fi->offset);
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
  switch (ty->toBasetype()->ty) {
  case TY::Tcomplex32: // treat it as {float32, float32}
    result.fields[0].ty = pointerTo(Type::tfloat32);
    result.fields[1].ty = pointerTo(Type::tfloat32);
    result.fields[0].offset = baseOffset;
    result.fields[1].offset = baseOffset + 4;
    result.length = 2;
    break;
  case TY::Tcomplex64: // treat it as {float64, float64}
    result.fields[0].ty = pointerTo(Type::tfloat64);
    result.fields[1].ty = pointerTo(Type::tfloat64);
    result.fields[0].offset = baseOffset;
    result.fields[1].offset = baseOffset + 8;
    result.length = 2;
    break;
  default:
    if (ty->toBasetype()->size() > 8) {
      // field larger than XLEN and FLEN
      result.length = -1;
      break;
    }
    result.fields[0].ty = ty->toBasetype();
    result.fields[0].offset = baseOffset;
    result.length = 1;
    break;
  }
  return result;
}

bool requireHardfloatRewrite(Type *ty) {
  if (!ty->toBasetype()->isTypeStruct())
    return false;
  auto result = visitStructFields(ty, 0);
  if (result.length <= 0)
    return false;
  if (result.length == 1)
    return result.fields[0].ty->isfloating();
  return result.fields[0].ty->isfloating() || result.fields[1].ty->isfloating();
}

struct HardfloatRewrite : ABIRewrite {
  LLValue *put(DValue *dv, bool, bool) override {
    // realign fields
    // FIXME: no need to alloc an extra buffer in many conditions
    const auto flat = visitStructFields(dv->type, 0);
    LLType *asType = type(dv->type, flat);
    const unsigned alignment = getABITypeAlign(asType);
    assert(dv->isLVal());
    LLValue *address = DtoLVal(dv);
    LLValue *buffer =
        DtoRawAlloca(asType, alignment, ".HardfloatRewrite_arg_storage");
    for (unsigned i = 0; i < (unsigned)flat.length; ++i) {
      DtoMemCpy(DtoGEP(asType, buffer, 0, i),
                DtoGEP1(getI8Type(), address, flat.fields[i].offset),
                DtoConstSize_t(flat.fields[i].ty->size()));
    }
    return DtoLoad(asType, buffer, ".HardfloatRewrite_arg");
  }
  LLValue *getLVal(Type *dty, LLValue *v) override {
    // inverse operation of method "put"
    const auto flat = visitStructFields(dty, 0);
    LLType *asType = type(dty, flat);
    const unsigned alignment = DtoAlignment(dty);
    LLValue *buffer = DtoAllocaDump(v, asType, getABITypeAlign(asType),
                                    ".HardfloatRewrite_param");
    LLValue *ret = DtoRawAlloca(DtoType(dty), alignment,
                                ".HardfloatRewrite_param_storage");
    for (unsigned i = 0; i < (unsigned)flat.length; ++i) {
      DtoMemCpy(DtoGEP1(getI8Type(), ret, flat.fields[i].offset),
                DtoGEP(asType, buffer, 0, i),
                DtoConstSize_t(flat.fields[i].ty->size()));
    }
    return ret;
  }
  LLType *type(Type *ty, const FlattenedFields &flat) {
    if (flat.length == 1) {
      return LLStructType::get(gIR->context(), {DtoType(flat.fields[0].ty)},
                               false);
    }
    assert(flat.length == 2);
    LLType *t[2];
    for (unsigned i = 0; i < 2; ++i) {
      t[i] = flat.fields[i].ty->isfloating()
                 ? DtoType(flat.fields[i].ty)
                 : LLIntegerType::get(gIR->context(),
                                      flat.fields[i].ty->size() * 8);
    }
    return LLStructType::get(gIR->context(), {t[0], t[1]}, false);
  }
  LLType *type(Type *ty) override { return type(ty, visitStructFields(ty, 0)); }
};
} // anonymous namespace

struct RISCV64TargetABI : TargetABI {
private:
  HardfloatRewrite hardfloatRewrite;
  IndirectByvalRewrite indirectByvalRewrite;
  Integer2Rewrite integer2Rewrite;
  IntegerRewrite integerRewrite;

public:
  Type *vaListType() override {
    // va_list is void*
    return pointerTo(Type::tvoid);
  }
  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref()) {
      return false;
    }
    Type *rt = tf->next->toBasetype();
    if (!rt->size())
      return false;
    if (!isPOD(rt))
      return true;
    return rt->size() > 16;
  }
  bool passByVal(TypeFunction *, Type *t) override {
    if (!t->size())
      return false;
    if (t->toBasetype()->ty == TY::Tcomplex80) {
      // rewrite it later to bypass the RVal problem
      return false;
    }
    return t->size() > 16;
  }
  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!fty.ret->byref) {
      if (!skipReturnValueRewrite(fty)) {
        if (!fty.ret->byref && isPOD(fty.ret->type) &&
            requireHardfloatRewrite(fty.ret->type)) {
          // rewrite here because we should not apply this to variadic arguments
          hardfloatRewrite.applyTo(*fty.ret);
        } else {
          rewriteArgument(fty, *fty.ret);
        }
      }
    }

    for (auto arg : fty.args) {
      if (!arg->byref && isPOD(arg->type) &&
          requireHardfloatRewrite(arg->type)) {
        // rewrite here because we should not apply this to variadic arguments
        hardfloatRewrite.applyTo(*arg);
      } else {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    if (arg.byref) {
      return;
    }

    Type *ty = arg.type->toBasetype();
    if (ty->ty == TY::Tcomplex80) {
      // {real, real} should be passed in memory
      indirectByvalRewrite.applyTo(arg);
      return;
    }

    if (!isPOD(arg.type)) {
      // non-PODs should be passed in memory
      indirectByvalRewrite.applyTo(arg);
      return;
    }

    if (isAggregate(ty) && ty->size() && ty->size() <= 16) {
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
TargetABI *getRISCV64TargetABI() { return new RISCV64TargetABI(); }
