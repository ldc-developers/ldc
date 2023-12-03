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
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Optional.h"
#else
#include <optional>
namespace llvm {
template <typename T> using Optional = std::optional<T>;
}
#endif

class Dsymbol;
class Type;

bool isFromLDC_DCompute(Dsymbol *sym);
bool isFromLDC_OpenCL(Dsymbol *sym);

struct DcomputePointer {
  int addrspace;
  Type *type;
  DcomputePointer(int as, Type *ty) : addrspace(as), type(ty) {}
  LLType *toLLVMType(bool translate) {
    auto llType = DtoType(type);
    int as = addrspace;
    if (translate)
      as = gIR->dcomputetarget->mapping[as];
    return llType->getPointerTo(as);
  }
};
llvm::Optional<DcomputePointer> toDcomputePointer(StructDeclaration *sd);
