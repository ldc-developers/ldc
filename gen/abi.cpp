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
#include "dmd/target.h"
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

// A Homogeneous Floating-point/Vector Aggregate (HFA/HVA) is an ARM/AArch64
// concept that consists of up to 4 elements of the same floating point/vector
// type. It is the aggregate final data layout that matters so nested structs,
// unions, and sarrays can result in an HFA.
//
// simple HFAs: struct F1 {float f;}  struct D4 {double a,b,c,d;}
// interesting HFA: struct {F1[2] vals; float weight;}

namespace {
// Recursive helper.
// Returns -1 if the type isn't suited as HFVA (element) or incompatible to the
// specified fundamental type, otherwise the number of consumed elements of that
// fundamental type.
// If `fundamentalType` is null, it is set on the first occasion and then left
// untouched.
int getNestedHFVA(Type *t, LLType *&fundamentalType) {
  t = t->toBasetype();
  int N = 0;

  if (auto tarray = t->isTypeSArray()) {
    N = getNestedHFVA(tarray->nextOf(), fundamentalType);
    return N < 0 ? N : N * tarray->dim->toUInteger(); // => T[0] may return 0
  }

  if (auto tstruct = t->isTypeStruct()) {
    // check each field recursively and set fundamentalType
    bool isEmpty = true;
    for (VarDeclaration *field : tstruct->sym->fields) {
      int field_N = getNestedHFVA(field->type, fundamentalType);
      if (field_N < 0)
        return field_N;
      if (field_N > 0) // might be 0 for empty static array
        isEmpty = false;
    }

    // an empty struct (no fields or only empty static arrays) is an undefined
    // byte, i.e., no HFVA
    if (isEmpty)
      return -1;

    // due to possibly overlapping fields (for unions and nested anonymous
    // unions), use the overall struct size to determine N
    const auto structSize = t->size();
    const auto fundamentalSize = fundamentalType->getPrimitiveSizeInBits() / 8;
    assert(structSize % fundamentalSize == 0);
    return structSize / fundamentalSize;
  }

  LLType *this_ft = nullptr;
  if (auto tvector = t->isTypeVector()) {
    this_ft = DtoType(tvector);
    N = 1;
  } else if (t->isfloating()) {
    auto tfloat = t;
    N = 1;
    if (t->iscomplex()) {
      N = 2;
      switch (t->ty) {
      case Tcomplex32:
        tfloat = Type::tfloat32;
        break;
      case Tcomplex64:
        tfloat = Type::tfloat64;
        break;
      case Tcomplex80:
        tfloat = Type::tfloat80;
        break;
      default:
        llvm_unreachable("Unexpected complex floating point type");
      }
    }
    this_ft = DtoType(tfloat);
  } else {
    return -1; // reject all other types
  }

  if (!fundamentalType) {
    fundamentalType = this_ft; // initialize fundamentalType
  } else if (fundamentalType != this_ft) {
    return -1; // incompatible fundamental types, reject
  }

  return N;
}
}

bool TargetABI::isHFVA(Type *t, llvm::Type **rewriteType, int maxElements) {
  if (!isAggregate(t) || !isPOD(t))
    return false;

  LLType *fundamentalType = nullptr;
  const int N = getNestedHFVA(t, fundamentalType);
  if (N < 1 || N > maxElements)
    return false;

  if (rewriteType)
    *rewriteType = LLArrayType::get(fundamentalType, N);

  return true;
}

TypeTuple *TargetABI::getArgTypes(Type *t) {
  // try to reuse cached argTypes of StructDeclarations
  if (auto ts = t->toBasetype()->isTypeStruct()) {
    auto sd = ts->sym;
    if (sd && sd->sizeok == SIZEOKdone)
      return sd->argTypes;
  }

  return target.toArgTypes(t);
}

LLType *TargetABI::getRewrittenArgType(Type *t, TypeTuple *argTypes) {
  if (!argTypes || argTypes->arguments->empty() ||
      (argTypes->arguments->length == 1 &&
       argTypes->arguments->front()->type == t)) {
    return nullptr; // don't rewrite
  }

  auto &args = *argTypes->arguments;
  return args.length == 1
             ? DtoType(args[0]->type)
             : LLStructType::get(gIR->context(), {DtoType(args[0]->type),
                                                  DtoType(args[1]->type)});
}

LLType *TargetABI::getRewrittenArgType(Type *t) {
  return getRewrittenArgType(t, getArgTypes(t));
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

bool TargetABI::isExternD(TypeFunction *tf) {
  return tf->linkage == LINKd && tf->parameterList.varargs != VARARGvariadic;
}

//////////////////////////////////////////////////////////////////////////////

bool TargetABI::reverseExplicitParams(TypeFunction *tf) {
  // Required by druntime for extern(D), except for `, ...`-style variadics.
  return isExternD(tf) && tf->parameterList.length() > 1;
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
