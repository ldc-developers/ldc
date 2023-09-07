//===-- abi.cpp -----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"

#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/target.h"
#include "gen/abi/targets.h"
#include "gen/abi/generic.h"
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
  llvm::Type *t = DtoType(dty);
  return DtoLoad(t, DtoBitCast(getLVal(dty, v), t->getPointerTo()));
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

bool TargetABI::isHVA(Type *t, int maxNumElements, LLType **hvaType) {
  Type *rewriteType = nullptr;
  if (::isHFVA(t, maxNumElements, &rewriteType) &&
      rewriteType->nextOf()->ty == TY::Tvector) {
    if (hvaType)
      *hvaType = DtoType(rewriteType);
    return true;
  }
  return false;
}

//////////////////////////////////////////////////////////////////////////////

TypeTuple *TargetABI::getArgTypes(Type *t) {
  // try to reuse cached argTypes of StructDeclarations
  if (auto ts = t->toBasetype()->isTypeStruct()) {
    auto sd = ts->sym;
    if (sd->sizeok == Sizeok::done)
      return sd->argTypes;
  }

  return target.toArgTypes(t);
}

LLType *TargetABI::getRewrittenArgType(Type *t, TypeTuple *argTypes) {
  if (!argTypes || argTypes->arguments->empty() ||
      (argTypes->arguments->length == 1 &&
       argTypes->arguments->front()->type->equivalent(t))) {
    return nullptr; // don't rewrite
  }

  auto &args = *argTypes->arguments;
  assert(args.length <= 2);
  return args.length == 1
             ? DtoType(args[0]->type)
             : LLStructType::get(gIR->context(), {DtoType(args[0]->type),
                                                  DtoType(args[1]->type)});
}

LLType *TargetABI::getRewrittenArgType(Type *t) {
  return getRewrittenArgType(t, getArgTypes(t));
}

//////////////////////////////////////////////////////////////////////////////

bool TargetABI::isAggregate(Type *t) {
  TY ty = t->toBasetype()->ty;
  // FIXME: dynamic arrays can currently not be rewritten as they are used
  //        by runtime functions, for which we don't apply the rewrites yet
  //        when calling them
  return ty == TY::Tstruct || ty == TY::Tsarray ||
         /*ty == TY::Tarray ||*/ ty == TY::Tdelegate || t->iscomplex();
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
  if (t->ty != TY::Tstruct)
    return true;
  StructDeclaration *sd = static_cast<TypeStruct *>(t)->sym;
  return sd->isPOD() && !(excludeStructsWithCtor && hasCtor(sd));
}

bool TargetABI::canRewriteAsInt(Type *t, bool include64bit) {
  auto size = t->toBasetype()->size();
  return size == 1 || size == 2 || size == 4 || (include64bit && size == 8);
}

bool TargetABI::isExternD(TypeFunction *tf) {
  return tf->linkage == LINK::d && tf->parameterList.varargs != VARARGvariadic;
}

bool TargetABI::skipReturnValueRewrite(IrFuncTy &fty) {
  if (fty.ret->byref)
    return true;

  auto ty = fty.ret->type->toBasetype()->ty;
  return ty == TY::Tvoid || ty == TY::Tnoreturn;
}

//////////////////////////////////////////////////////////////////////////////

llvm::CallingConv::ID TargetABI::callingConv(TypeFunction *tf, bool) {
  return tf->parameterList.varargs == VARARGvariadic
             ? static_cast<llvm::CallingConv::ID>(llvm::CallingConv::C)
             : callingConv(tf->linkage);
}

llvm::CallingConv::ID TargetABI::callingConv(FuncDeclaration *fdecl) {
  auto tf = fdecl->type->isTypeFunction();
  assert(tf);
  return callingConv(tf, fdecl->needThis() || fdecl->isNested());
}

//////////////////////////////////////////////////////////////////////////////

bool TargetABI::preferPassByRef(Type *t) {
  // simple base heuristic: use a ref for all types > 2 machine words
  return t->size() > 2 * target.ptrsize;
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
    DtoMemCpy(DtoType(dest->type), llDest, DtoLVal(src));
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
    if (tf->isref()) {
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
    return getMIPS64TargetABI(global.params.targetTriple->isArch64Bit());
  case llvm::Triple::riscv64:
    return getRISCV64TargetABI();
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
#if LDC_LLVM_VER >= 1600
  case llvm::Triple::loongarch64:
    return getLoongArch64TargetABI();
#endif // LDC_LLVM_VER >= 1600
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

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *ty = arg.type->toBasetype();
    if (ty->ty != TY::Tstruct) {
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
      if (rt->ty == TY::Tstruct) {
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
