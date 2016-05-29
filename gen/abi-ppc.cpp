//===-- abi-ppc64.cpp -----------------------------------------------------===//
//
//                         LDC ? the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 32/64 bit big-endian PowerPC targets.
//
// The System V Application Binary Interface PowerPC Processor Supplement can be
// found here:
// http://refspecs.linuxfoundation.org/elf/elfspec_ppc.pdf
//
// The PowerOpen 64bit ABI can be found here:
// http://refspecs.linuxfoundation.org/ELF/ppc64/PPC-elf64abi-1.9.html
// http://refspecs.linuxfoundation.org/ELF/ppc64/PPC-elf64abi-1.9.pdf
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-ppc.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct PPCTargetABI : TargetABI {
  ExplicitByvalRewrite byvalRewrite;
  IntegerRewrite integerRewrite;
  const bool Is64Bit;

  explicit PPCTargetABI(const bool Is64Bit) : Is64Bit(Is64Bit) {}

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref) {
      return false;
    }

    // Return structs and static arrays on the stack. The latter is needed
    // because otherwise LLVM tries to actually return the array in a number
    // of physical registers, which leads, depending on the target, to
    // either horrendous codegen or backend crashes.
    Type *rt = tf->next->toBasetype();
    return (rt->ty == Tstruct || rt->ty == Tsarray);
  }

  bool passByVal(Type *t) override {
    TY ty = t->toBasetype()->ty;
    return ty == Tstruct || ty == Tsarray;
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    // EXPLICIT PARAMETERS
    for (auto arg : fty.args) {
      if (!arg->byref) {
        rewriteArgument(fty, *arg);
      }
    }
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *ty = arg.type->toBasetype();

    if (ty->ty == Tstruct || ty->ty == Tsarray) {
      if (canRewriteAsInt(ty, Is64Bit)) {
        if (!IntegerRewrite::isObsoleteFor(arg.ltype)) {
          arg.rewrite = &integerRewrite;
          arg.ltype = integerRewrite.type(arg.type);
        }
      } else {
        // these types are passed byval:
        // the caller allocates a copy and then passes a pointer to the copy
        arg.rewrite = &byvalRewrite;
        arg.ltype = byvalRewrite.type(arg.type);

        // the copy is treated as a local variable of the callee
        // hence add the NoAlias and NoCapture attributes
        arg.attrs.clear()
            .add(LLAttribute::NoAlias)
            .add(LLAttribute::NoCapture)
            .addAlignment(byvalRewrite.alignment(arg.type));
      }
    }
  }
};

// The public getter for abi.cpp
TargetABI *getPPCTargetABI(bool Is64Bit) {
  return new PPCTargetABI(Is64Bit);
}
