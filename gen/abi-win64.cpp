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
#include "id.h"

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
  ExplicitByvalRewrite byvalRewrite;
  IntegerRewrite integerRewrite;
  MSVCLongDoubleRewrite longDoubleRewrite;

  static bool isMagicCppLongDoubleStruct(Type *t) {
    return t->ty == Tstruct &&
           static_cast<TypeStruct *>(t)->sym->ident == Id::__c_long_double;
  }

  static bool realIs80bits() {
    return !global.params.targetTriple.isWindowsMSVCEnvironment();
  }

  // Returns true if the D type is passed byval (the callee getting a pointer
  // to a dedicated hidden copy).
  static bool isPassedWithByvalSemantics(Type *t) {
    return
        // * aggregates which can NOT be rewritten as integers
        //   (size > 64 bits or not a power of 2)
        (isAggregate(t) && !canRewriteAsInt(t)) ||
        // * 80-bit real and ireal
        (realIs80bits() && (t->ty == Tfloat80 || t->ty == Timaginary80));
  }

public:
  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref) {
      return false;
    }

    Type *rt = tf->next->toBasetype();

    // * let LLVM return 80-bit real/ireal on the x87 stack, for DMD compliance
    if (realIs80bits() && (rt->ty == Tfloat80 || rt->ty == Timaginary80)) {
      return false;
    }

    // * all POD types <= 64 bits and of a size that is a power of 2
    //   (incl. 2x32-bit cfloat) are returned in a register (RAX, or
    //   XMM0 for single float/ifloat/double/idouble)
    // * all other types are returned via struct-return (sret)
    return (rt->ty == Tstruct && !isPOD(rt, tf->linkage)) ||
           isPassedWithByvalSemantics(rt);
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
    if (!fty.ret->byref && fty.ret->type->toBasetype()->ty != Tvoid) {
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

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *t = arg.type->toBasetype();

    if (isPassedWithByvalSemantics(t)) {
      // these types are passed byval:
      // the caller allocates a copy and then passes a pointer to the copy
      arg.rewrite = &byvalRewrite;

      // the copy is treated as a local variable of the callee
      // hence add the NoAlias and NoCapture attributes
      arg.attrs.clear()
          .add(LLAttribute::NoAlias)
          .add(LLAttribute::NoCapture)
          .addAlignment(byvalRewrite.alignment(arg.type));
    } else if (isMagicCppLongDoubleStruct(t)) {
      arg.rewrite = &longDoubleRewrite;
    } else if (isAggregate(t) && canRewriteAsInt(t) &&
               !IntegerRewrite::isObsoleteFor(arg.ltype)) {
      arg.rewrite = &integerRewrite;
    }

    if (arg.rewrite) {
      LLType *originalLType = arg.ltype;
      arg.ltype = arg.rewrite->type(arg.type, arg.ltype);

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
