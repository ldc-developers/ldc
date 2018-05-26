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
//
//===----------------------------------------------------------------------===//

#include "ldcbindings.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-aarch64.h"

struct AArch64TargetABI : TargetABI {
  HFAToArray hfaToArray;
  CompositeToArray64 compositeToArray64;
  IntegerRewrite integerRewrite;

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    return passByVal(rt);
  }

  bool passByVal(Type *t) override {
    t = t->toBasetype();
    return t->ty == Tsarray ||
           (t->ty == Tstruct && t->size() > 16 && !isHFA((TypeStruct *)t));
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    Type *retTy = fty.ret->type->toBasetype();
    if (!fty.ret->byref && retTy->ty == Tstruct) {
      // Rewrite HFAs only because union HFAs are turned into IR types that are
      // non-HFA and messes up register selection
      if (isHFA((TypeStruct *)retTy, &fty.ret->ltype)) {
        hfaToArray.applyTo(*fty.ret, fty.ret->ltype);
      } else {
        integerRewrite.applyTo(*fty.ret);
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
    // FIXME
    Type *ty = arg.type->toBasetype();

    if (ty->ty == Tstruct || ty->ty == Tsarray) {
      // Rewrite HFAs only because union HFAs are turned into IR types that are
      // non-HFA and messes up register selection
      if (ty->ty == Tstruct && isHFA((TypeStruct *)ty, &arg.ltype)) {
        hfaToArray.applyTo(arg, arg.ltype);
      } else {
        compositeToArray64.applyTo(arg);
      }
    }
  }

  /**
  * The AACPS64 uses a special native va_list type:
  *
  * typedef struct __va_list {
  *     void *__stack; // next stack param
  *     void *__gr_top; // end of GP arg reg save area
  *     void *__vr_top; // end of FP/SIMD arg reg save area
  *     int __gr_offs; // offset from __gr_top to next GP register arg
  *     int __vr_offs; // offset from __vr_top to next FP/SIMD register arg
  * } va_list;
  *
  * In druntime, the struct is defined as core.stdc.stdarg.__va_list; the
  * actually used core.stdc.stdarg.va_list type is a raw char* pointer though to
  * achieve byref semantics.
  * This requires a little bit of compiler magic in the following
  * implementations.
  */

  LLType *getValistType() {
    LLType *intType = LLType::getInt32Ty(gIR->context());
    LLType *voidPointerType = getVoidPtrType();

    std::vector<LLType *> parts;      // struct __va_list {
    parts.push_back(voidPointerType); //   void *__stack;
    parts.push_back(voidPointerType); //   void *__gr_top;
    parts.push_back(voidPointerType); //   void *__vr_top;
    parts.push_back(intType);         //   int __gr_offs;
    parts.push_back(intType);         //   int __vr_offs; };

    return LLStructType::get(gIR->context(), parts);
  }

  LLValue *prepareVaStart(DLValue *ap) override {
    // Since the user only created a char* pointer (ap) on the stack before
    // invoking va_start, we first need to allocate the actual __va_list struct
    // and set `ap` to its address.
    LLValue *valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
    DtoStore(valistmem,
             DtoBitCast(DtoLVal(ap), getPtrToType(valistmem->getType())));
    // Pass a i8* pointer to the actual struct to LLVM's va_start intrinsic.
    return DtoBitCast(valistmem, getVoidPtrType());
  }

  void vaCopy(DLValue *dest, DValue *src) override {
    // Analog to va_start, we first need to allocate a new __va_list struct on
    // the stack and set `dest` to its address.
    LLValue *valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
    DtoStore(valistmem,
             DtoBitCast(DtoLVal(dest), getPtrToType(valistmem->getType())));
    // Then fill the new struct with a bitcopy of the source struct.
    // `src` is a char* pointer to the source struct.
    DtoMemCpy(valistmem, DtoRVal(src));
  }

  LLValue *prepareVaArg(DLValue *ap) override {
    // Pass a i8* pointer to the actual __va_list struct to LLVM's va_arg
    // intrinsic.
    return DtoRVal(ap);
  }

  Type *vaListType() override {
    // We need to pass the actual va_list type for correct mangling. Simply
    // using TypeIdentifier here is a bit wonky but works, as long as the name
    // is actually available in the scope (this is what DMD does, so if a better
    // solution is found there, this should be adapted).
    static const llvm::StringRef ident = "__va_list";
    return (createTypeIdentifier(Loc(), Identifier::idPool(ident.data(), ident.size())));
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    return "objc_msgSend";
  }
};

// The public getter for abi.cpp
TargetABI *getAArch64TargetABI() { return new AArch64TargetABI(); }
