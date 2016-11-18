//===-- gen/abi-spirv.cpp ---------------------------------------*- C++ -*-===//
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

struct SPIRVTargetABI : TargetABI {
  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                    FuncDeclaration *fdecl = nullptr) override {
    if (hasKernelAttr(fdecl))
      return llvm::CallingConv::SPIR_KERNEL;
    else
      return llvm::CallingConv::SPIR_FUNC;
  }
  bool passByVal(Type *t) override {
    // TODO: Do some field testing to figure out the most optimal cutoff.
    // 16 means that a float4 will be passed in registers. Must be greater than
    // 8 so that e.g. { float* } does not get the byVal attribute so there are no
    // double pointers, as this is disallowed in earlier versions of OpenCL.
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

TargetABI *createSPIRVABI() { return new SPIRVTargetABI(); }