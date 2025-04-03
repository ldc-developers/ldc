//===-- abi-systemz.cpp
//-----------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit big-endian IBM Z targets.
//
// The IBM s390x ELF ABI can be found here:
// https://github.com/IBM/s390x-abi
//===----------------------------------------------------------------------===//

#include "dmd/identifier.h"
#include "dmd/nspace.h"
#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

using namespace dmd;

struct SimpleHardfloatRewrite : ABIRewrite {
  Type *getFirstFieldType(Type *ty) {
    if (auto ts = ty->toBasetype()->isTypeStruct()) {
      assert(ts->sym->fields.size() == 1);
      auto *subField = ts->sym->fields[0];
      if (subField->type->isfloating()) {
        return subField->type;
      }
      return nullptr;
    }
    return nullptr;
  }

  LLValue *put(DValue *dv, bool, bool) override {
    const auto flat = getFirstFieldType(dv->type);
    LLType *asType = DtoType(flat);
    assert(dv->isLVal());
    LLValue *flatGEP = DtoGEP1(asType, DtoLVal(dv), 0U);
    LLValue *flatValue = DtoLoad(asType, flatGEP, ".HardfloatRewrite_arg");
    return flatValue;
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    // inverse operation of method "put"
    LLValue *insertedValue = DtoInsertValue(llvm::UndefValue::get(DtoType(dty)), v, 0);
    return DtoAllocaDump(insertedValue, dty, ".HardfloatRewrite_param_storage");
  }

  LLType *type(Type *ty) override { return DtoType(getFirstFieldType(ty)); }

  bool shouldApplyRewrite(Type *ty) {
    if (auto ts = ty->toBasetype()->isTypeStruct()) {
      return ts->sym->fields.size() == 1 &&
             ts->sym->fields[0]->type->isfloating();
    }
    return false;
  }
};

struct StructSimpleFlattenRewrite : BaseBitcastABIRewrite {
  LLType *type(Type *ty) override {
    const size_t type_size = size(ty);
    // "A struct or a union of 1, 2, 4, or 8 bytes"
    switch (type_size) {
    case 1:
      return LLType::getInt8Ty(gIR->context());
    case 2:
      return LLType::getInt16Ty(gIR->context());
    case 4:
      return LLType::getInt32Ty(gIR->context());
    case 8:
      return LLType::getInt64Ty(gIR->context());
    default:
      return DtoType(ty);
    }
  }
};

struct SystemZTargetABI : TargetABI {
  IndirectByvalRewrite indirectByvalRewrite{};
  StructSimpleFlattenRewrite structSimpleFlattenRewrite{};
  SimpleHardfloatRewrite simpleHardfloatRewrite{};

  explicit SystemZTargetABI() {}

  bool isSystemZVaList(Type *t) {
    // look for a __va_list struct in a `std` C++ namespace
    if (auto ts = t->isTypeStruct()) {
      auto sd = ts->sym;
      if (strcmp(sd->ident->toChars(), "__va_list_tag") == 0) {
        if (auto ns = sd->parent->isNspace()) {
          return strcmp(ns->toChars(), "std") == 0;
        }
      }
    }

    return false;
  }

  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref()) {
      return false;
    }
    Type *rt = tf->next->toBasetype();
    if (rt->ty == TY::Tstruct) {
      return true;
    }
    if (rt->isTypeVector() && size(rt) > 16) {
      return true;
    }
    return shouldPassByVal(tf->next);
  }

  bool passByVal(TypeFunction *, Type *t) override {
    // LLVM's byval attribute is not compatible with the SystemZ ABI
    // due to how SystemZ's stack is setup
    return false;
  }

  bool shouldPassByVal(Type *t) {
    if (t->ty == TY::Tstruct && size(t) <= 8) {
      return false;
    }
    // "A struct or union of any other size, a complex type, an __int128, a long
    // double, a _Decimal128, or a vector whose size exceeds 16 bytes"
    if (size(t) > 16 || t->iscomplex() || t->isimaginary()) {
      return true;
    }
    if (t->ty == TY::Tint128 || t->ty == TY::Tcomplex80) {
      return true;
    }
    return DtoIsInMemoryOnly(t);
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!fty.ret->byref) {
      rewriteArgument(fty, *fty.ret);
    }

    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    if (!isPOD(arg.type) || shouldPassByVal(arg.type)) {
      // non-PODs should be passed in memory
      indirectByvalRewrite.applyTo(arg);
      return;
    }
    Type *ty = arg.type->toBasetype();
    // compiler magic: pass va_list args implicitly by reference
    if (isSystemZVaList(ty)) {
      arg.byref = true;
      arg.ltype = arg.ltype->getPointerTo();
      return;
    }
    // integer types less than 64-bits should be extended to 64 bits
    if (ty->isintegral() &&
        !(ty->ty == TY::Tstruct || ty->ty == TY::Tsarray ||
          ty->ty == TY::Tvector) &&
        size(ty) < 8) {
      arg.attrs.addAttribute(ty->isunsigned() ? LLAttribute::ZExt
                                              : LLAttribute::SExt);
    }
    if (ty->isTypeStruct()) {
      if (simpleHardfloatRewrite.shouldApplyRewrite(ty)) {
        simpleHardfloatRewrite.applyTo(arg);
      } else if (size(ty) <= 8) {
        structSimpleFlattenRewrite.applyToIfNotObsolete(arg);
      }
    }
  }

  Type *vaListType() override {
    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a
    // better solution is found there, this should be adapted).
    return dmd::pointerTo(
        TypeIdentifier::create(Loc(), Identifier::idPool("__va_list_tag")));
  }

  /**
   * The SystemZ ABI (like AMD64) uses a special native va_list type -
   * a 32-bytes struct passed by reference.
   * In druntime, the struct is aliased as object.__va_list_tag; the actually
   * used core.stdc.stdarg.va_list type is a __va_list_tag* pointer though to
   * achieve byref semantics. This requires a little bit of compiler magic in
   * the following implementations.
   */

  LLType *getValistType() {
    LLType *longType = LLType::getInt64Ty(gIR->context());
    LLType *pointerType = getOpaquePtrType();

    std::vector<LLType *> parts;  // struct __va_list_tag {
    parts.push_back(longType);    //   long __gpr;
    parts.push_back(longType);    //   long __fpr;
    parts.push_back(pointerType); //   void *__overflow_arg_area;
    parts.push_back(pointerType); //   void *__reg_save_area; }

    return LLStructType::get(gIR->context(), parts);
  }

  LLValue *prepareVaStart(DLValue *ap) override {
    // Since the user only created a __va_list_tag* pointer (ap) on the stack
    // before invoking va_start, we first need to allocate the actual
    // __va_list_tag struct and set `ap` to its address.
    LLValue *valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
    DtoStore(valistmem, DtoLVal(ap));
    // Pass an opaque pointer to the actual struct to LLVM's va_start intrinsic.
    return valistmem;
  }

  void vaCopy(DLValue *dest, DValue *src) override {
    // Analog to va_start, we first need to allocate a new __va_list_tag struct
    // on the stack and set `dest` to its address.
    LLValue *valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
    DtoStore(valistmem, DtoLVal(dest));
    // Then fill the new struct with a bitcopy of the source struct.
    // `src` is a __va_list_tag* pointer to the source struct.
    DtoMemCpy(getValistType(), valistmem, DtoRVal(src));
  }

  LLValue *prepareVaArg(DLValue *ap) override {
    // Pass an opaque pointer to the actual __va_list_tag struct to LLVM's
    // va_arg intrinsic.
    return DtoRVal(ap);
  }
};

// The public getter for abi.cpp
TargetABI *getSystemZTargetABI() { return new SystemZTargetABI(); }
