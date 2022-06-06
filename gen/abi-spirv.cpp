//===-- gen/abi-spirv.cpp ---------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/dcompute/druntime.h"
#include "gen/uda.h"
#include "dmd/declaration.h"
#include "gen/tollvm.h"
#include "gen/dcompute/abi-rewrites.h"

struct SPIRVTargetABI : TargetABI {
  DComputePointerRewrite pointerRewite;
  llvm::CallingConv::ID callingConv(LINK) override {
    llvm_unreachable("expected FuncDeclaration overload to be used");
  }
  llvm::CallingConv::ID callingConv(FuncDeclaration *fdecl) override {
    return hasKernelAttr(fdecl) ? llvm::CallingConv::SPIR_KERNEL
                                : llvm::CallingConv::SPIR_FUNC;
  }
  bool passByVal(TypeFunction *, Type *t) override {
    t = t->toBasetype();
    return ((t->ty == TY::Tsarray || t->ty == TY::Tstruct) && t->size() > 64);
  }
  void rewriteFunctionType(IrFuncTy &fty) override {
    for (auto arg : fty.args) {
      if (!arg->byref)
        rewriteArgument(fty, *arg);
    }
    if (!skipReturnValueRewrite(fty))
      rewriteArgument(fty, *fty.ret);
  }
  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref())
      return false;
    Type *retty = tf->next->toBasetype();
    if (retty->ty == TY::Tsarray)
      return true;
    else if (auto st = retty->isTypeStruct())
      return !toDcomputePointer(st->sym);
    else
      return false;
  }
  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *ty = arg.type->toBasetype();
    llvm::Optional<DcomputePointer> ptr;
    if (ty->ty == TY::Tstruct &&
        (ptr = toDcomputePointer(static_cast<TypeStruct *>(ty)->sym))) {
      pointerRewite.applyTo(arg);
    }
  }
  // There are no exceptions at all, so no need for unwind tables.
  bool needsUnwindTables() override {
    return false;
  }
};

TargetABI *createSPIRVABI() { return new SPIRVTargetABI(); }
