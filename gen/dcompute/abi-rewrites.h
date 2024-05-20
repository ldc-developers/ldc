//===-- gen/dcompute/abi-rewrites.h - dcompute ABI rewrites -----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains ABI rewrites for the dcompute targets SPIR-V and NVPTX
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/abi/generic.h"

struct DComputePointerRewrite : ABIRewrite {
  LLValue *put(DValue *v, bool isLValueExp, bool) override {
    LLValue *address = DtoLVal(v);
    return DtoLoad(type(v->type), address, ".DComputePointerRewrite_arg");
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    LLValue *mem = DtoAlloca(dty, ".DComputePointerRewrite_param_storage");
    DtoStore(v, mem);
    return mem;
  }

  LLType *type(Type *t) override {
    auto ptr = toDcomputePointer(static_cast<TypeStruct *>(t)->sym);
    return ptr->toLLVMType(true);
  }
};
