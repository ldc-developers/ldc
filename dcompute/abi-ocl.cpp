//===-- dcompute/abi-ocl.cpp -----------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#include "dcompute/abi.h"
#include "gen/uda.h"
struct OCLTargetABI : TargetABI {
    llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                      FuncDeclaration *fdecl = nullptr) override {
        //Is this ever called with fdecl == null ?
        if(!fdecl)
            return llvm::CallingConv::C;
        if(hasKernelAttr(fdecl))
            return llvm::CallingConv::SPIR_KERNEL;
        else
            return llvm::CallingConv::SPIR_FUNC;
    }
    bool passByVal(Type *t) override {
        //TODO: understand what this does
        return false;
    }
    void rewriteFunctionType(TypeFunction *t, IrFuncTy &fty) override {
        //do nothing
    }
    bool returnInArg(TypeFunction *tf) override
    {
        return false;
    }
};

TargetABI* createOCLABI()
{
    return new OCLTargetABI();
}

