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

struct NVPTXTargetABI : TargetABI {
  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                    FuncDeclaration *fdecl = nullptr) override {
    if (hasKernelAttr(fdecl))
      return llvm::CallingConv::PTX_Kernel;
    else
      return llvm::CallingConv::PTX_Device;
  }
  bool passByVal(Type *t) override {
    return t->size() <= 16; //enough for a float4
  }
  void rewriteFunctionType(TypeFunction *t, IrFuncTy &fty) override {
    // Do nothing.
  }
  bool returnInArg(TypeFunction *tf) override { return false; }
};

TargetABI *createNVPTXABI() { return new NVPTXTargetABI(); }