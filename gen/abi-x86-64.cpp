//===-- abi-x86-64.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// extern(C) implements the C calling convention for x86-64, as found in
// http://www.x86-64.org/documentation/abi-0.99.pdf
//
// Note:
//   Where a discrepancy was found between llvm-gcc and the ABI documentation,
//   llvm-gcc behavior was used for compatibility (after it was verified that
//   regular gcc has the same behavior).
//
// LLVM gets it right for most types, but complex numbers, structs and static
// arrays need some help. To make sure it gets those right we essentially
// bitcast these types to a type to which LLVM assigns the appropriate
// registers (using DMD's toArgTypes() machinery), and pass that instead.
// Structs that are required to be passed in memory are marked with the ByVal
// attribute to ensure no part of them ends up in registers when only a subset
// of the desired registers are available.
//
//===----------------------------------------------------------------------===//

#include "gen/abi-x86-64.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/identifier.h"
#include "dmd/ldcbindings.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-x86-64.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include <cassert>
#include <map>
#include <string>
#include <utility>

namespace {
// Structs, static arrays and cfloats may be rewritten to exploit registers.
// This function returns the rewritten type, or null if no transformation is
// needed.
LLType *getAbiType(Type *ty) {
  // First, check if there's any need of a transformation:
  if (!(ty->ty == Tcomplex32 || ty->ty == Tstruct || ty->ty == Tsarray)) {
    return nullptr; // Nothing to do
  }

  // Okay, we may need to transform. Figure out a canonical type:
  llvm::SmallVector<Type *, 2> argTypes;

  // try to reuse cached arg{1,2}type of StructDeclarations
  if (ty->ty == Tstruct) {
    const auto sd = static_cast<TypeStruct *>(ty)->sym;
    if (sd && sd->sizeok == SIZEOKdone) {
      if (!sd->arg1type) {
        return nullptr; // don't rewrite
      }
      argTypes.push_back(sd->arg1type);
      if (sd->arg2type) {
        argTypes.push_back(sd->arg2type);
      }
    }
  }

  if (argTypes.empty()) {
    TypeTuple *tuple = target.toArgTypes(ty);
    if (!tuple || tuple->arguments->empty()) {
      return nullptr; // don't rewrite
    }
    for (auto param : *tuple->arguments) {
      argTypes.push_back(param->type);
    }
  }

  LLType *abiTy = nullptr;
  if (argTypes.size() == 1) {
    abiTy = DtoType(argTypes[0]);
  } else {
    abiTy = LLStructType::get(gIR->context(),
                              {DtoType(argTypes[0]), DtoType(argTypes[1])});
  }

  return abiTy;
}

bool passByVal(Type *ty) {
  TypeTuple *argTypes = target.toArgTypes(ty);
  if (!argTypes) {
    return false;
  }

  return argTypes->arguments->empty(); // empty => cannot be passed in registers
}

struct RegCount {
  char int_regs, sse_regs;

  RegCount() : int_regs(6), sse_regs(8) {}

  explicit RegCount(LLType *ty) : int_regs(0), sse_regs(0) {
    if (LLStructType *structTy = isaStruct(ty)) {
      for (unsigned i = 0; i < structTy->getNumElements(); ++i) {
        RegCount elementRegCount(structTy->getElementType(i));
        int_regs += elementRegCount.int_regs;
        sse_regs += elementRegCount.sse_regs;
      }
    } else if (LLArrayType *arrayTy = isaArray(ty)) {
      char N = static_cast<char>(arrayTy->getNumElements());
      RegCount elementRegCount(arrayTy->getElementType());
      int_regs = N * elementRegCount.int_regs;
      sse_regs = N * elementRegCount.sse_regs;
    } else if (ty->isIntegerTy() || ty->isPointerTy()) {
      ++int_regs;
    } else if (ty->isFloatingPointTy() || ty->isVectorTy()) {
      // X87 reals are passed on the stack
      if (!ty->isX86_FP80Ty()) {
        ++sse_regs;
      }
    } else {
      unsigned sizeInBits = gDataLayout->getTypeSizeInBits(ty);
      IF_LOG Logger::cout() << "SysV RegCount: assuming 1 GP register for type "
                            << *ty << " (" << sizeInBits << " bits)\n";
      assert(sizeInBits > 0 && sizeInBits <= 64);
      ++int_regs;
    }

    assert(int_regs + sse_regs <= 2);
  }

  enum SubtractionResult {
    ArgumentFitsIn,
    ArgumentWouldFitInPartially,
    ArgumentDoesntFitIn
  };

  SubtractionResult trySubtract(const IrFuncTyArg &arg) {
    const RegCount wanted(arg.ltype);

    const bool anyRegAvailable = (wanted.int_regs > 0 && int_regs > 0) ||
                                 (wanted.sse_regs > 0 && sse_regs > 0);
    if (!anyRegAvailable) {
      return ArgumentDoesntFitIn;
    }

    if (int_regs < wanted.int_regs || sse_regs < wanted.sse_regs) {
      return ArgumentWouldFitInPartially;
    }

    int_regs -= wanted.int_regs;
    sse_regs -= wanted.sse_regs;

    return ArgumentFitsIn;
  }
};
}

/**
 * This type performs the actual struct/cfloat rewriting by simply storing to
 * memory so that it's then readable as the other type (i.e., bit-casting).
 */
struct X86_64_C_struct_rewrite : ABIRewrite {
  LLValue *put(DValue *v, bool, bool) override {
    LLValue *address = getAddressOf(v);

    LLType *abiTy = getAbiType(v->type->toBasetype());
    assert(abiTy && "Why are we rewriting a non-rewritten type?");

    return loadFromMemory(address, abiTy, ".X86_64_C_struct_rewrite_putResult");
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return DtoAllocaDump(v, dty, ".X86_64_C_struct_rewrite_dump");
  }

  LLType *type(Type *t) override { return getAbiType(t->toBasetype()); }
};

/**
 * This type is used to force LLVM to pass a LL aggregate in memory,
 * on the function parameters stack. We need this to prevent LLVM
 * from passing an aggregate partially in registers, partially in
 * memory.
 * This is achieved by passing a pointer to the aggregate and using
 * the byval LLVM attribute.
 */
struct ImplicitByvalRewrite : ABIRewrite {
  LLValue *put(DValue *v, bool isLValueExp, bool isLastArgExp) override {
    if (isLValueExp && !isLastArgExp && v->isLVal()) {
      // copy to avoid visibility of potential side effects of later argument
      // expressions
      return DtoAllocaDump(v, ".lval_copy_for_ImplicitByvalRewrite");
    }
    return getAddressOf(v);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override { return v; }

  LLType *type(Type *t) override { return DtoPtrToType(t); }

  void applyTo(IrFuncTyArg &arg, LLType *finalLType = nullptr) override {
    ABIRewrite::applyTo(arg, finalLType);
    arg.attrs.addAttribute(LLAttribute::ByVal);
    if (auto alignment = DtoAlignment(arg.type))
      arg.attrs.addAlignmentAttr(alignment);
  }
};

struct X86_64TargetABI : TargetABI {
  X86_64_C_struct_rewrite struct_rewrite;
  ImplicitByvalRewrite byvalRewrite;
  IndirectByvalRewrite indirectByvalRewrite;

  bool returnInArg(TypeFunction *tf, bool needsThis) override;

  bool passByVal(TypeFunction *tf, Type *t) override;

  void rewriteFunctionType(IrFuncTy &fty) override;
  void rewriteVarargs(IrFuncTy &fty, std::vector<IrFuncTyArg *> &args) override;
  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override;
  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg, RegCount &regCount);

  LLValue *prepareVaStart(DLValue *ap) override;

  void vaCopy(DLValue *dest, DValue *src) override;

  LLValue *prepareVaArg(DLValue *ap) override;

  Type *vaListType() override;

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override;

private:
  LLType *getValistType();
  RegCount &getRegCount(IrFuncTy &fty) {
    return reinterpret_cast<RegCount &>(fty.tag);
  }
};

// The public getter for abi.cpp
TargetABI *getX86_64TargetABI() { return new X86_64TargetABI; }

bool X86_64TargetABI::returnInArg(TypeFunction *tf, bool) {
  if (tf->isref) {
    return false;
  }

  Type *rt = tf->next->toBasetype();
  return ::passByVal(rt);
}

bool X86_64TargetABI::passByVal(TypeFunction *tf, Type *t) {
  // indirectly by-value for non-POD args
  if (!isPOD(t))
    return false;

  return ::passByVal(t->toBasetype());
}

void X86_64TargetABI::rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) {
  llvm_unreachable("Please use the other overload explicitly.");
}

void X86_64TargetABI::rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg,
                                      RegCount &regCount) {
  LLType *originalLType = arg.ltype;
  Type *t = arg.type->toBasetype();

  // indirectly by-value for non-POD args
  if (!isPOD(t)) {
    indirectByvalRewrite.applyTo(arg);
    if (regCount.int_regs > 0) {
      regCount.int_regs--;
    }

    return;
  }

  LLType *abiTy = getAbiType(t);
  if (abiTy && !LLTypeMemoryLayout::typesAreEquivalent(abiTy, originalLType)) {
    IF_LOG {
      Logger::println("Rewriting argument type %s", t->toChars());
      LOG_SCOPE;
      Logger::cout() << *originalLType << " => " << *abiTy << '\n';
    }

    struct_rewrite.applyTo(arg, abiTy);
  }

  if (regCount.trySubtract(arg) == RegCount::ArgumentWouldFitInPartially) {
    // pass the LL aggregate with byval attribute to prevent LLVM from passing
    // it partially in registers, partially in memory
    assert(originalLType->isAggregateType());
    IF_LOG Logger::cout() << "Passing byval to prevent register/memory mix: "
                          << arg.type->toChars() << " (" << *originalLType
                          << ")\n";
    byvalRewrite.applyTo(arg);
  }
}

void X86_64TargetABI::rewriteFunctionType(IrFuncTy &fty) {
  RegCount &regCount = getRegCount(fty);
  regCount = RegCount(); // initialize

  // RETURN VALUE
  if (!fty.ret->byref && fty.ret->type->toBasetype()->ty != Tvoid) {
    Logger::println("x86-64 ABI: Transforming return type");
    LOG_SCOPE;
    RegCount dummy;
    rewriteArgument(fty, *fty.ret, dummy);
  }

  // IMPLICIT PARAMETERS
  if (fty.arg_sret) {
    regCount.int_regs--;
  }
  if (fty.arg_this || fty.arg_nest) {
    regCount.int_regs--;
  }
  if (fty.arg_objcSelector) {
    regCount.int_regs--;
  }
  if (fty.arg_arguments) {
    regCount.int_regs -= 2; // dynamic array
  }

  // EXPLICIT PARAMETERS
  Logger::println("x86-64 ABI: Transforming argument types");
  LOG_SCOPE;

  int begin = 0, end = fty.args.size(), step = 1;
  if (fty.reverseParams) {
    begin = end - 1;
    end = -1;
    step = -1;
  }
  for (int i = begin; i != end; i += step) {
    IrFuncTyArg &arg = *fty.args[i];

    if (arg.byref) {
      if (!arg.isByVal() && regCount.int_regs > 0) {
        regCount.int_regs--;
      }

      continue;
    }

    rewriteArgument(fty, arg, regCount);
  }

  // regCount (fty.tag) is now in the state after all implicit & formal args,
  // ready to serve as initial state for each vararg call site, see below
}

void X86_64TargetABI::rewriteVarargs(IrFuncTy &fty,
                                     std::vector<IrFuncTyArg *> &args) {
  // use a dedicated RegCount copy for each call site and initialize it with
  // fty.tag
  RegCount regCount = getRegCount(fty);

  for (auto arg : args) {
    if (!arg->byref) { // don't rewrite ByVal arguments
      rewriteArgument(fty, *arg, regCount);
    }
  }
}

/**
 * The System V AMD64 ABI uses a special native va_list type - a 24-bytes struct
 * passed by reference.
 * In druntime, the struct is defined as object.__va_list_tag; the actually used
 * core.stdc.stdarg.va_list type is a __va_list_tag* pointer though to achieve
 * byref semantics.
 * This requires a little bit of compiler magic in the following
 * implementations.
 */

LLType *X86_64TargetABI::getValistType() {
  LLType *uintType = LLType::getInt32Ty(gIR->context());
  LLType *voidPointerType = getVoidPtrType();

  std::vector<LLType *> parts;      // struct __va_list_tag {
  parts.push_back(uintType);        //   uint gp_offset;
  parts.push_back(uintType);        //   uint fp_offset;
  parts.push_back(voidPointerType); //   void* overflow_arg_area;
  parts.push_back(voidPointerType); //   void* reg_save_area; }

  return LLStructType::get(gIR->context(), parts);
}

LLValue *X86_64TargetABI::prepareVaStart(DLValue *ap) {
  // Since the user only created a __va_list_tag* pointer (ap) on the stack before
  // invoking va_start, we first need to allocate the actual __va_list_tag struct
  // and set `ap` to its address.
  LLValue *valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
  DtoStore(valistmem,
           DtoBitCast(DtoLVal(ap), getPtrToType(valistmem->getType())));
  // Pass a i8* pointer to the actual struct to LLVM's va_start intrinsic.
  return DtoBitCast(valistmem, getVoidPtrType());
}

void X86_64TargetABI::vaCopy(DLValue *dest, DValue *src) {
  // Analog to va_start, we first need to allocate a new __va_list_tag struct on
  // the stack and set `dest` to its address.
  LLValue *valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
  DtoStore(valistmem,
           DtoBitCast(DtoLVal(dest), getPtrToType(valistmem->getType())));
  // Then fill the new struct with a bitcopy of the source struct.
  // `src` is a __va_list_tag* pointer to the source struct.
  DtoMemCpy(valistmem, DtoRVal(src));
}

LLValue *X86_64TargetABI::prepareVaArg(DLValue *ap) {
  // Pass a i8* pointer to the actual __va_list_tag struct to LLVM's va_arg
  // intrinsic.
  return DtoBitCast(DtoRVal(ap), getVoidPtrType());
}

Type *X86_64TargetABI::vaListType() {
  // We need to pass the actual va_list type for correct mangling. Simply
  // using TypeIdentifier here is a bit wonky but works, as long as the name
  // is actually available in the scope (this is what DMD does, so if a better
  // solution is found there, this should be adapted).
  return createTypeIdentifier(Loc(), Identifier::idPool("__va_list_tag"))
      ->pointerTo();
}

const char *X86_64TargetABI::objcMsgSendFunc(Type *ret,
                                             IrFuncTy &fty) {
  // see objc/message.h for objc_msgSend selection rules
  if (fty.arg_sret) {
    return "objc_msgSend_stret";
  }
  if (ret) {
    // complex long double return
    if (ret->ty == Tcomplex80) {
      return "objc_msgSend_fp2ret";
    }
    // long double return
    if (ret->ty == Tfloat80 || ret->ty == Timaginary80) {
      return "objc_msgSend_fpret";
    }
  }
  return "objc_msgSend";
}
