//===-- gen/abi-nvptx.cpp ---------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/uda.h"
#include "ddmd/declaration.h"
#include "tollvm.h"

struct NVPTXTargetABI : TargetABI {
  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                    FuncDeclaration *fdecl = nullptr) override {
    assert(fdecl);
    if (hasKernelAttr(fdecl))
      return llvm::CallingConv::PTX_Kernel;
    else
      return llvm::CallingConv::PTX_Device;
  }
  bool passByVal(Type *t) override {
    return DtoIsInMemoryOnly(t);
  }
  void rewriteFunctionType(TypeFunction *t, IrFuncTy &fty) override {
    // Do nothing.
  }
  bool returnInArg(TypeFunction *tf) override {
    return !tf->isref && DtoIsInMemoryOnly(tf->next);
  }
};

TargetABI *createNVPTXABI() { return new NVPTXTargetABI(); }