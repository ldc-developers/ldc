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

#include "mtype.h"
#include "declaration.h"
#include "aggregate.h"

#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/dvalue.h"
#include "gen/llvmhelpers.h"
#include "gen/abi.h"
#include "gen/abi-win64.h"
#include "gen/abi-generic.h"
#include "ir/irfunction.h"

#include <cassert>
#include <string>
#include <utility>

struct Win64TargetABI : TargetABI {
private:
  const bool isMSVC;
  ExplicitByvalRewrite byvalRewrite;
  IntegerRewrite integerRewrite;

  bool isX87(Type *t) const {
    return !isMSVC // 64-bit reals for MSVC targets
           && (t->ty == Tfloat80 || t->ty == Timaginary80);
  }

  bool passPointerToHiddenCopy(Type *t, bool isReturnValue, LINK linkage) const {
    // Pass magic C++ structs directly as LL aggregate with a single i32/double
    // element, which LLVM handles as if it was a scalar.
    if (isMagicCppStruct(t))
      return false;

    // 80-bit real/ireal:
    // * returned on the x87 stack (for DMD inline asm compliance and what LLVM
    //   defaults to)
    // * passed by ref to hidden copy
    if (isX87(t))
      return !isReturnValue;

    const bool isMSVCpp = isMSVC && linkage == LINKcpp;

    if (isReturnValue) {
      // MSVC++ enforces sret for non-PODs, incl. aggregates with ctors
      // (which by itself doesn't make it a non-POD for D).
      const bool excludeStructsWithCtor = isMSVCpp;
      if (!isPOD(t, excludeStructsWithCtor))
        return true;
    } else {
      // Contrary to return values, POD-ness is ignored for arguments.
      // MSVC++ seems to enforce by-ref passing only for aggregates with
      // copy ctor (incl. `= delete`).
      if (isMSVCpp && t->ty == Tstruct) {
        StructDeclaration *sd = static_cast<TypeStruct *>(t)->sym;
        assert(sd);
        if (sd->postblit)
          return true;
      }
    }

    // Remaining aggregates which can NOT be rewritten as integers (size > 8
    // bytes or not a power of 2) are passed by ref to hidden copy.
    return isAggregate(t) && !canRewriteAsInt(t);
  }

  LINK &getLinkage(IrFuncTy &fty) {
    return reinterpret_cast<LINK &>(fty.tag);
  }

public:
  Win64TargetABI()
      : isMSVC(global.params.targetTriple->isWindowsMSVCEnvironment()) {}

  llvm::CallingConv::ID callingConv(LINK l, TypeFunction *tf = nullptr,
                                    FuncDeclaration *fd = nullptr) override {
    // Use the vector calling convention for extern(D) (except for variadics)
    // => let LLVM pass vectors in registers instead of passing a ref to a
    // hidden copy (both cases handled by LLVM automatically for LL vectors
    // which we don't rewrite).
    return l == LINKd && !(tf && tf->varargs == 1)
               ? llvm::CallingConv::X86_VectorCall
               : llvm::CallingConv::C;
  }

  std::string mangleFunctionForLLVM(std::string name, LINK l) override {
    if (l == LINKd) {
      // Prepend a 0x1 byte to prevent LLVM from applying vectorcall/stdcall
      // mangling: _D… => _D…@<paramssize>
      name.insert(name.begin(), '\1');
    }
    return name;
  }

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref)
      return false;

    // * all POD types of a power-of-2 size <= 8 bytes (incl. 2x32-bit cfloat)
    //   are returned in a register (RAX, or XMM0 for single float/ifloat/
    //   double/idouble)
    // * 80-bit real/ireal are returned on the x87 stack
    // * all other types are returned via sret
    Type *rt = tf->next->toBasetype();
    return passPointerToHiddenCopy(rt, /*isReturnValue=*/true, tf->linkage);
  }

  bool passByVal(Type *t) override {
    // LLVM's byval attribute is not compatible with the Win64 ABI
    return false;
  }

  bool passThisBeforeSret(TypeFunction *tf) override {
    // required by MSVC++
    return tf->linkage == LINKcpp;
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    // store linkage in fty.tag as required by rewriteArgument() later
    getLinkage(fty) = tf->linkage;

    // return value
    const auto rt = fty.ret->type->toBasetype();
    if (!fty.ret->byref && rt->ty != Tvoid) {
      rewrite(fty, *fty.ret, /*isReturnValue=*/true);
    }

    // explicit parameters
    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }

    // extern(D): reverse parameter order for non variadics, for DMD-compliance
    if (tf->linkage == LINKd && tf->varargs != 1 && fty.args.size() > 1) {
      fty.reverseParams = true;
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

    if (passPointerToHiddenCopy(t, isReturnValue, getLinkage(fty))) {
      // the caller allocates a hidden copy and passes a pointer to that copy
      arg.rewrite = &byvalRewrite;

      // the copy is treated as a local variable of the callee
      // hence add the NoAlias and NoCapture attributes
      arg.attrs.clear()
          .add(LLAttribute::NoAlias)
          .add(LLAttribute::NoCapture)
          .addAlignment(byvalRewrite.alignment(arg.type));
    } else if (isAggregate(t) && canRewriteAsInt(t) && !isMagicCppStruct(t) &&
               !IntegerRewrite::isObsoleteFor(arg.ltype)) {
      arg.rewrite = &integerRewrite;
    }

    if (arg.rewrite) {
      LLType *originalLType = arg.ltype;
      arg.ltype = arg.rewrite->type(arg.type);

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
