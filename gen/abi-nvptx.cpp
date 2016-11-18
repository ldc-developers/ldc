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
    if (hasKernelAttr(fdecl))
      return llvm::CallingConv::PTX_Kernel;
    else
      return llvm::CallingConv::PTX_Device;
  }
  bool passByVal(Type *t) override {
    // TODO: Do some field testing to figure out the most optimal cutoff.
    // 16 means that a float4 will be passed in registers.
    return t->size() > 16 && DtoIsInMemoryOnly(t);
  }
  void rewriteFunctionType(TypeFunction *t, IrFuncTy &fty) override {
    // Do nothing.
  }
  bool returnInArg(TypeFunction *tf) override {
    // Never use sret because we don't know what addrspace the implicit pointer
    // should address.
    return false;
  }
};

TargetABI *createNVPTXABI() { return new NVPTXTargetABI(); }