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

#ifndef LDC_GEN_DCOMPUTE_ABI_REWRITES_H
#define LDC_GEN_DCOMPUTE_ABI_REWRITES_H

#include "gen/abi.h"
#include "gen/dcompute/druntime.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/tollvm.h"

struct DComputePointerRewrite : ABIRewrite {
  LLType *type(Type *t) override {
    auto ptr = toDcomputePointer(static_cast<TypeStruct *>(t)->sym);
    return ptr->toLLVMType(true);
  }
  LLValue *getLVal(Type *dty, LLValue *v) override {
    // TODO: Is this correct?
    return DtoAllocaDump(v, this->type(dty));
  }
  LLValue *put(DValue *dv, bool, bool) override {
    LLValue *address = getAddressOf(dv);
    LLType *t = this->type(dv->type);
    return loadFromMemory(address, t);
  }
};
#endif
