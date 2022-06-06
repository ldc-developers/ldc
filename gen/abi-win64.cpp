//===-- abi-win64.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// extern(C) implements the C calling convention for x86-64 on Windows, see
// http://msdn.microsoft.com/en-us/library/7kcdt6fy%28v=vs.110%29.aspx
//
//===----------------------------------------------------------------------===//

#include "gen/abi-win64.h"

#include "dmd/mtype.h"
#include "dmd/declaration.h"
#include "dmd/aggregate.h"
#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include <cassert>
#include <string>
#include <utility>

struct Win64TargetABI : TargetABI {
private:
  const bool isMSVC;
  IndirectByvalRewrite byvalRewrite;
  IntegerRewrite integerRewrite;
  HFVAToArray hfvaToArray;

  bool isX87(Type *t) const {
    return !isMSVC // 64-bit reals for MSVC targets
           && (t->ty == TY::Tfloat80 || t->ty == TY::Timaginary80);
  }

  bool passPointerToHiddenCopy(Type *t, bool isReturnValue,
                               TypeFunction *tf) const {
    return passPointerToHiddenCopy(t, isReturnValue, tf->linkage, tf);
  }

  bool passPointerToHiddenCopy(Type *t, bool isReturnValue, LINK linkage,
                               TypeFunction *tf = nullptr) const {
    // 80-bit real/ireal:
    // * returned on the x87 stack (for DMD inline asm compliance and what LLVM
    //   defaults to)
    // * passed by ref to hidden copy
    if (isX87(t))
      return !isReturnValue;

    const bool isMSVCpp = isMSVC && linkage == LINK::cpp;

    // Handle non-PODs:
    if (isReturnValue) {
      // Enforce sret for non-PODs.
      // MSVC++ additionally enforces it for all structs with ctors.
      if (!isPOD(t, isMSVCpp))
        return true;
    } else {
      // MSVC++ seems to enforce by-ref passing only for structs with
      // copy ctor (incl. `= delete`).
      if (isMSVCpp) {
        if (t->ty == TY::Tstruct) {
          StructDeclaration *sd = static_cast<TypeStruct *>(t)->sym;
          assert(sd);
          if (sd->postblit || sd->hasCopyCtor)
            return true;
        }
      }
      // non-MSVC++: pass all non-PODs by ref to hidden copy
      else if (!isPOD(t)) {
        return true;
      }
    }

    // __vectorcall: Homogeneous Vector Aggregates are passed in registers
    const bool isD = tf ? isExternD(tf) : linkage == LINK::d;
    if (isD && isHVA(t, hfvaToArray.maxElements))
      return false;

    // Remaining aggregates which can NOT be rewritten as integers (size > 8
    // bytes or not a power of 2) are passed by ref to hidden copy.
    // LDC-specific exceptions: slices and delegates are left alone (as non-
    // rewritten IR structs) and passed/returned as 2 separate args => passed in
    // up to 2 GP registers and returned in RAX & RDX.
    return isAggregate(t) && !canRewriteAsInt(t) && t->ty != TY::Tarray &&
           t->ty != TY::Tdelegate;
  }

public:
  Win64TargetABI()
      : isMSVC(global.params.targetTriple->isWindowsMSVCEnvironment()),
        hfvaToArray(4) {}

  // Use the vector calling convention for extern(D) (except for variadics) =>
  // let LLVM pass vectors in registers instead of passing a ref to a hidden
  // copy (both cases handled by LLVM automatically for LL vectors which we
  // don't rewrite).
  llvm::CallingConv::ID callingConv(LINK l) override {
    return l == LINK::d ? llvm::CallingConv::X86_VectorCall
                        : llvm::CallingConv::C;
  }

  std::string mangleFunctionForLLVM(std::string name, LINK l) override {
    if (l == LINK::d) {
      // Prepend a 0x1 byte to prevent LLVM from applying vectorcall/stdcall
      // mangling: _D… => _D…@<paramssize>
      name.insert(name.begin(), '\1');
    }
    return name;
  }

  bool returnInArg(TypeFunction *tf, bool needsThis) override {
    if (tf->isref())
      return false;

    Type *rt = tf->next->toBasetype();

    // for non-static member functions, MSVC++ enforces sret for all structs
    if (isMSVC && tf->linkage == LINK::cpp && needsThis &&
        rt->ty == TY::Tstruct) {
      return true;
    }

    // * all POD types of a power-of-2 size <= 8 bytes (incl. 2x32-bit cfloat)
    //   are returned in a register (RAX, or XMM0 for single float/ifloat/
    //   double/idouble)
    // * 80-bit real/ireal are returned on the x87 stack
    // * LDC-specific: slices and delegates are returned in RAX & RDX
    // * for extern(D), vectors and Homogeneous Vector Aggregates are returned
    //   in SIMD register(s)
    // * all other types are returned via sret
    return passPointerToHiddenCopy(rt, /*isReturnValue=*/true, tf);
  }

  // Prefer a ref if the POD cannot be passed in a register, i.e., if
  // IndirectByvalRewrite would be applied.
  bool preferPassByRef(Type *t) override {
    return passPointerToHiddenCopy(t->toBasetype(), false, LINK::d);
  }

  bool passByVal(TypeFunction *, Type *) override {
    // LLVM's byval attribute is not compatible with the Win64 ABI
    return false;
  }

  bool passThisBeforeSret(TypeFunction *tf) override {
    // required by MSVC++
    return tf->linkage == LINK::cpp;
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    // return value
    if (!skipReturnValueRewrite(fty)) {
      rewrite(fty, *fty.ret, /*isReturnValue=*/true);
    }

    // explicit parameters
    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteVarargs(IrFuncTy &fty,
                      std::vector<IrFuncTyArg *> &args) override {
    for (auto arg : args) {
      rewriteArgument(fty, *arg);

      if (arg->rewrite == &byvalRewrite) {
        // mark the vararg as being passed byref to prevent DtoCall() from
        // passing the dereferenced pointer, i.e., just pass the pointer
        arg->byref = true;
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    rewrite(fty, arg, /*isReturnValue=*/false);
  }

  void rewrite(IrFuncTy &fty, IrFuncTyArg &arg, bool isReturnValue) {
    Type *t = arg.type->toBasetype();
    LLType *originalLType = arg.ltype;

    if (passPointerToHiddenCopy(t, isReturnValue, fty.type)) {
      // the caller allocates a hidden copy and passes a pointer to that copy
      byvalRewrite.applyTo(arg);
    } else if (isExternD(fty.type) && isHVA(t, hfvaToArray.maxElements, &arg.ltype)) {
      // rewrite Homogeneous Vector Aggregates as static array of vectors
      hfvaToArray.applyTo(arg, arg.ltype);
    } else if (isAggregate(t) && canRewriteAsInt(t)) {
      integerRewrite.applyToIfNotObsolete(arg);
    }

    if (arg.rewrite) {
      IF_LOG {
        Logger::println("Rewriting argument type %s", t->toChars());
        LOG_SCOPE;
        Logger::cout() << *originalLType << " => " << *arg.ltype << '\n';
      }
    }
  }
};

// The public getter for abi.cpp
TargetABI *getWin64TargetABI() { return new Win64TargetABI; }
