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

#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dcompute/druntime.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"

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

struct DcomputeMetalScalarRewrite : ABIRewrite {
  LLType *type(Type* t) override {
    // XXXX: Scalar variables are stored in the constant memory space for Metal GPU
    return llvm::PointerType::get(gIR->context(), 2/*Constant Memory space*/);
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    return v;
  }

  LLValue *put(DValue *v, bool isLValueExp, bool) override {
    auto value = DtoRVal(v);
    return value;
  }
};
