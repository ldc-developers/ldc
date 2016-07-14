//===-- dcompute/ab-cuda.cpp ------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#include "dcompute/abi-cuda.h"
#include "gen/uda.h"
struct CUDATargetABI : TargetABI {
    llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                      FuncDeclaration *fdecl = nullptr) override {
        //Is this ever called with fdecl == null ?
        if(!fdecl)
            return llvm::CallingConv::C;
        if(hasKernelAttr(fdecl))
            return llvm::CallingConv::PTX_Kernel;
        else
            return llvm::CallingConv::PTX_Device;
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

TargetABI* createCudaABI()
{
    return new CUDATargetABI();
}

