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

  // Returns true if the D type is passed byval (the callee getting a pointer
  // to a dedicated hidden copy).
  bool isPassedWithByvalSemantics(Type *t) const {
    return
        // * aggregates which can NOT be rewritten as integers
        //   (size > 8 bytes or not a power of 2)
        (isAggregate(t) && !canRewriteAsInt(t)) ||
        // * 80-bit real and ireal
        isX87(t);
  }

public:
  Win64TargetABI()
      : isMSVC(global.params.targetTriple->isWindowsMSVCEnvironment()) {}

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref)
      return false;

    Type *rt = tf->next->toBasetype();

    // let LLVM return
    // * magic C++ structs directly as LL aggregate with a single i32/double
    //   element, which LLVM handles as if it was a scalar
    // * 80-bit real/ireal on the x87 stack, for DMD inline asm compliance
    if (isMagicCppStruct(rt) || isX87(rt))
      return false;

    // force sret for non-POD structs
    const bool excludeStructsWithCtor = (isMSVC && tf->linkage == LINKcpp);
    if (!isPOD(rt, excludeStructsWithCtor))
      return true;

    // * all POD types of a power-of-2 size <= 8 bytes (incl. 2x32-bit cfloat)
    //   are returned in a register (RAX, or XMM0 for single float/ifloat/
    //   double/idouble)
    // * all other types are returned via sret
    return isPassedWithByvalSemantics(rt);
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
    // return value
    const auto rt = fty.ret->type->toBasetype();
    if (!fty.ret->byref && rt->ty != Tvoid) {
      // 80-bit real/ireal are returned on the x87 stack (but passed in memory)
      if (!isX87(rt))
        rewriteArgument(fty, *fty.ret);
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
    Type *t = arg.type->toBasetype();

    if (isMagicCppStruct(t)) {
      // pass directly as LL aggregate
    } else if (isPassedWithByvalSemantics(t)) {
      // these types are passed byval:
      // the caller allocates a copy and then passes a pointer to the copy
      arg.rewrite = &byvalRewrite;

      // the copy is treated as a local variable of the callee
      // hence add the NoAlias and NoCapture attributes
      arg.attrs.clear()
          .add(LLAttribute::NoAlias)
          .add(LLAttribute::NoCapture)
          .addAlignment(byvalRewrite.alignment(arg.type));
    } else if (isAggregate(t) && canRewriteAsInt(t) &&
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
