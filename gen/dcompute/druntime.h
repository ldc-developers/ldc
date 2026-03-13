//===-- gen/dcompute/druntime.h ---------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
// Functionality related to ldc.dcompute in druntime
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/aggregate.h"
#include "dmd/mtype.h"
#include "gen/dcompute/target.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include <optional>

class Dsymbol;
class Type;

bool isFromLDC_DCompute(Dsymbol *sym);
bool isFromLDC_OpenCL(Dsymbol *sym);

struct DcomputePointer {
  int addrspace;
  Type *type;
  DcomputePointer(int as, Type *ty) : addrspace(as), type(ty) {}
  LLType *toLLVMType(bool translate) {
    DtoType(type);
    int as = addrspace;
    if (translate)
      as = gIR->dcomputetarget->mapping[as];
    return LLPointerType::get(getGlobalContext(), as);
  }
};
std::optional<DcomputePointer> toDcomputePointer(StructDeclaration *sd);
