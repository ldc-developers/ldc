//===-- abi.cpp -----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"

#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "gen/abi-aarch64.h"
#include "gen/abi-arm.h"
#include "gen/abi-generic.h"
#include "gen/abi-mips64.h"
#include "gen/abi-ppc.h"
#include "gen/abi-ppc64le.h"
#include "gen/abi-win64.h"
#include "gen/abi-x86.h"
#include "gen/abi-x86-64.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"
#include <algorithm>

// in dmd/argtypes_aarch64.d:
bool isHFVA(Type *t, int maxNumElements, Type **rewriteType);

//////////////////////////////////////////////////////////////////////////////

llvm::Value *ABIRewrite::getRVal(Type *dty, LLValue *v) {
  return DtoLoad(DtoBitCast(getLVal(dty, v), DtoType(dty)->getPointerTo()));
}

//////////////////////////////////////////////////////////////////////////////

void ABIRewrite::applyTo(IrFuncTyArg &arg, LLType *finalLType) {
  arg.rewrite = this;
  arg.ltype = finalLType ? finalLType : this->type(arg.type);
}

//////////////////////////////////////////////////////////////////////////////

LLValue *ABIRewrite::getAddressOf(DValue *v) {
  if (v->isLVal())
    return DtoLVal(v);
  return DtoAllocaDump(v, ".getAddressOf_dump");
}

//////////////////////////////////////////////////////////////////////////////

bool TargetABI::isHFVA(Type *t, int maxNumElements, LLType **hfvaType) {
  Type *rewriteType = nullptr;
  if (::isHFVA(t, maxNumElements, &rewriteType)) {
    if (hfvaType)
      *hfvaType = DtoType(rewriteType);
    return true;
  }
  return false;
}

bool TargetABI::isAggregate(Type *t) {
  TY ty = t->toBasetype()->ty;
  // FIXME: dynamic arrays can currently not be rewritten as they are used
  //        by runtime functions, for which we don't apply the rewrites yet
  //        when calling them
  return ty == Tstruct || ty == Tsarray ||
         /*ty == Tarray ||*/ ty == Tdelegate || t->iscomplex();
}

namespace {
bool hasCtor(StructDeclaration *s) {
  if (s->ctor)
    return true;
  for (VarDeclaration *field : s->fields) {
    Type *tf = field->type->baseElemOf();
    if (auto tstruct = tf->isTypeStruct()) {
      if (hasCtor(tstruct->sym))
        return true;
    }
  }
  return false;
}
}

bool TargetABI::isPOD(Type *t, bool excludeStructsWithCtor) {
  t = t->baseElemOf();
  if (t->ty != Tstruct)
    return true;
  StructDeclaration *sd = static_cast<TypeStruct *>(t)->sym;
  return sd->isPOD() && !(excludeStructsWithCtor && hasCtor(sd));
}

bool TargetABI::canRewriteAsInt(Type *t, bool include64bit) {
  auto size = t->toBasetype()->size();
  return size == 1 || size == 2 || size == 4 || (include64bit && size == 8);
}

//////////////////////////////////////////////////////////////////////////////

bool TargetABI::reverseExplicitParams(TypeFunction *tf) {
  // Required by druntime for extern(D), except for `, ...`-style variadics.
  return tf->linkage == LINKd &&
         tf->parameterList.varargs != VARARGvariadic &&
         tf->parameterList.length() > 1;
}

//////////////////////////////////////////////////////////////////////////////

void TargetABI::rewriteVarargs(IrFuncTy &fty,
                               std::vector<IrFuncTyArg *> &args) {
  for (auto arg : args) {
    if (!arg->byref) { // don't rewrite ByVal arguments
      rewriteArgument(fty, *arg);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////

LLValue *TargetABI::prepareVaStart(DLValue *ap) {
  // pass a i8* pointer to ap to LLVM's va_start intrinsic
  return DtoBitCast(DtoLVal(ap), getVoidPtrType());
}

//////////////////////////////////////////////////////////////////////////////

void TargetABI::vaCopy(DLValue *dest, DValue *src) {
  LLValue *llDest = DtoLVal(dest);
  if (src->isLVal()) {
    DtoMemCpy(llDest, DtoLVal(src));
  } else {
    DtoStore(DtoRVal(src), llDest);
  }
}

//////////////////////////////////////////////////////////////////////////////

LLValue *TargetABI::prepareVaArg(DLValue *ap) {
  // pass a i8* pointer to ap to LLVM's va_arg intrinsic
  return DtoBitCast(DtoLVal(ap), getVoidPtrType());
}

//////////////////////////////////////////////////////////////////////////////

Type *TargetABI::vaListType() {
  // char* is used by default in druntime.
  return Type::tchar->pointerTo();
}

//////////////////////////////////////////////////////////////////////////////

const char *TargetABI::objcMsgSendFunc(Type *ret, IrFuncTy &fty) {
  llvm_unreachable("Unknown Objective-C ABI");
}

//////////////////////////////////////////////////////////////////////////////

// Some reasonable defaults for when we don't know what ABI to use.
struct UnknownTargetABI : TargetABI {
  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref) {
      return false;
    }

    // Return structs and static arrays on the stack. The latter is needed
    // because otherwise LLVM tries to actually return the array in a number
    // of physical registers, which leads, depending on the target, to
    // either horrendous codegen or backend crashes.
    Type *rt = tf->next->toBasetype();
    return passByVal(tf, rt);
  }

  bool passByVal(TypeFunction *, Type *t) override {
    return DtoIsInMemoryOnly(t);
  }

  void rewriteFunctionType(IrFuncTy &) override {
    // why?
  }
};

//////////////////////////////////////////////////////////////////////////////

TargetABI *TargetABI::getTarget() {
  switch (global.params.targetTriple->getArch()) {
  case llvm::Triple::x86:
    return getX86TargetABI();
  case llvm::Triple::x86_64:
    if (global.params.targetTriple->isOSWindows()) {
      return getWin64TargetABI();
    } else {
      return getX86_64TargetABI();
    }
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    return getMIPS64TargetABI(global.params.is64bit);
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    return getPPCTargetABI(global.params.targetTriple->isArch64Bit());
  case llvm::Triple::ppc64le:
    return getPPC64LETargetABI();
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    return getAArch64TargetABI();
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    return getArmTargetABI();
  default:
    Logger::cout() << "WARNING: Unknown ABI, guessing...\n";
    return new UnknownTargetABI;
  }
}

//////////////////////////////////////////////////////////////////////////////

// A simple ABI for LLVM intrinsics.
struct IntrinsicABI : TargetABI {
  RemoveStructPadding remove_padding;

  bool returnInArg(TypeFunction *, bool) override { return false; }

  bool passByVal(TypeFunction *, Type *t) override { return false; }

  bool reverseExplicitParams(TypeFunction *) override { return false; }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *ty = arg.type->toBasetype();
    if (ty->ty != Tstruct) {
      return;
    }
    // TODO: Check that no unions are passed in or returned.

    LLType *abiTy = DtoUnpaddedStructType(arg.type);

    if (abiTy && abiTy != arg.ltype) {
      remove_padding.applyTo(arg, abiTy);
    }
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    if (!fty.arg_sret) {
      Type *rt = fty.ret->type->toBasetype();
      if (rt->ty == Tstruct) {
        Logger::println("Intrinsic ABI: Transforming return type");
        rewriteArgument(fty, *fty.ret);
      }
    }

    Logger::println("Intrinsic ABI: Transforming arguments");
    LOG_SCOPE;

    for (auto arg : fty.args) {
      IF_LOG Logger::cout() << "Arg: " << arg->type->toChars() << '\n';

      // Arguments that are in memory are of no interest to us.
      if (arg->byref) {
        continue;
      }

      rewriteArgument(fty, *arg);

      IF_LOG Logger::cout() << "New arg type: " << *arg->ltype << '\n';
    }
  }
};

TargetABI *TargetABI::getIntrinsic() {
  static IntrinsicABI iabi;
  return &iabi;
}
