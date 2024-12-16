//===-- gen/abi-nvptx.cpp ---------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/dcompute/druntime.h"
#include "gen/uda.h"
#include "dmd/declaration.h"
#include "gen/tollvm.h"
#include "gen/dcompute/abi-rewrites.h"

using namespace dmd;

struct NVPTXTargetABI : TargetABI {
  DComputePointerRewrite pointerRewite;
  llvm::CallingConv::ID callingConv(LINK l) override {
      assert(l == LINK::c);
      return llvm::CallingConv::PTX_Device;
  }
  llvm::CallingConv::ID callingConv(FuncDeclaration *fdecl) override {
    return hasKernelAttr(fdecl) ? llvm::CallingConv::PTX_Kernel
                                : llvm::CallingConv::PTX_Device;
  }
  bool passByVal(TypeFunction *, Type *t) override {
    return DtoIsInMemoryOnly(t) && isPOD(t) && size(t) > 64;
  }
  bool returnInArg(TypeFunction *tf, bool) override {
    return DtoIsInMemoryOnly(tf->next);
  }
  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    TargetABI::rewriteArgument(fty, arg);
    if (arg.rewrite)
      return;

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
