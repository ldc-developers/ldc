//===-- gen/abi-spirv.cpp ---------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/dcomputetypes.h"
#include "gen/uda.h"
#include "ddmd/declaration.h"
#include "gen/tollvm.h"

struct SPIRVTargetABI : TargetABI {
  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                    FuncDeclaration *fdecl = nullptr) override {
    assert(fdecl);
    if (hasKernelAttr(fdecl))
      return llvm::CallingConv::SPIR_KERNEL;
    else
      return llvm::CallingConv::SPIR_FUNC;
  }
  bool passByVal(Type *t) override {
    Type *typ = t->toBasetype();
    TY ty= typ->ty;
    if (ty == Tstruct)
      return !bool(toDcomputePointer(((TypeStruct*)typ)->sym));
    return (ty == Tsarray);  }
  void rewriteFunctionType(TypeFunction *t, IrFuncTy &fty) override {
    // Do nothing.
  }
  bool returnInArg(TypeFunction *tf) override {
    return !tf->isref && DtoIsInMemoryOnly(tf->next);
  }
};

TargetABI *createSPIRVABI() { return new SPIRVTargetABI(); }
