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
#include "ddmd/declaration.h"
#include "gen/tollvm.h"
#include "gen/dcompute/abi-rewrites.h"

struct SPIRVTargetABI : TargetABI {
  DComputePointerRewrite pointerRewite;
  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                    FuncDeclaration *fdecl = nullptr) override {
    assert(fdecl);
    if (hasKernelAttr(fdecl))
      return llvm::CallingConv::SPIR_KERNEL;
    else
      return llvm::CallingConv::SPIR_FUNC;
  }
  bool passByVal(Type *t) override {
    t = t->toBasetype();
    return ((t->ty == Tsarray || t->ty == Tstruct) && t->size() > 64);
  }
  void rewriteFunctionType(TypeFunction *t, IrFuncTy &fty) override {
    for (auto arg : fty.args) {
      if (!arg->byref)
        rewriteArgument(fty, *arg);
    }
  }
  bool returnInArg(TypeFunction *tf) override {
    return !tf->isref && DtoIsInMemoryOnly(tf->next);
  }
  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    Type *ty = arg.type->toBasetype();
    llvm::Optional<DcomputePointer> ptr;
    if (ty->ty == Tstruct &&
        (ptr = toDcomputePointer(static_cast<TypeStruct*>(ty)->sym)))
    {
      arg.rewrite = &pointerRewite;
      arg.ltype = pointerRewite.type(arg.type);
    }
  }
  // There are no exceptions at all, so no need for unwind tables.
  bool needsUnwindTables() override {
    return false;
  }
};

TargetABI *createSPIRVABI() { return new SPIRVTargetABI(); }
