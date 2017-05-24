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

#ifndef LDC_GEN_DCOMPUTE_DRUNTIME_H
#define LDC_GEN_DCOMPUTE_DRUNTIME_H

#include "ddmd/aggregate.h"
#include "ddmd/mtype.h"
#include "llvm/ADT/Optional.h"
#include "gen/dcompute/target.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"

class Dsymbol;
class Type;

bool isFromLDC_DCompute(Dsymbol *sym);

struct DcomputePointer {
  int addrspace;
  Type *type;
  DcomputePointer(int as, Type *ty) : addrspace(as), type(ty) {}
  LLType *toLLVMType(bool translate) {
    auto llType = DtoMemType(type);
    int as = addrspace;
    if (translate)
      as = gIR->dcomputetarget->mapping[as];
    return llType->getPointerElementType()->getPointerTo(as);
  }
};
llvm::Optional<DcomputePointer> toDcomputePointer(StructDeclaration *sd);
#endif
