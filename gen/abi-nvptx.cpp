//===-- gen/abi-nvptx.cpp ---------------------------------------*- C++ -*-===//
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

struct NVPTXTargetABI : TargetABI {
  DComputePointerRewrite pointerRewite;
  llvm::CallingConv::ID callingConv(LINK) override {
    llvm_unreachable("expected FuncDeclaration overload to be used");
  }
  llvm::CallingConv::ID callingConv(FuncDeclaration *fdecl) override {
    return hasKernelAttr(fdecl) ? llvm::CallingConv::PTX_Kernel
                                : llvm::CallingConv::PTX_Device;
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
  }
  bool returnInArg(TypeFunction *tf, bool) override {
    return !tf->isref() && DtoIsInMemoryOnly(tf->next);
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

TargetABI *createNVPTXABI() { return new NVPTXTargetABI(); }
