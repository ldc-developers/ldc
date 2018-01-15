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
#include "gen/dcompute/target.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "llvm/ADT/Optional.h"

class Dsymbol;
class Type;

bool isFromLDC_DCompute(Dsymbol *sym);

struct DcomputeAddrspacedType {
  unsigned addrspace;
  Type *type;
  Identifier *id; // Id::dcPointer or Id::dcVariable
  DcomputeAddrspacedType(int as, Type *ty, Identifier * _id) :
    addrspace(as), type(ty), id(_id) {}

  // Only used by the dcompute ABI rewrites so no need to check id
  LLType *toLLVMType(bool shouldTranslate) {
    auto llType = DtoType(type);
    unsigned as = addrspace;
    if (shouldTranslate)
      as = translate();
    return llType->getPointerTo(as);
  }
  unsigned translate()
  {
    return gIR ? gIR->dcomputetarget->mapping[addrspace] : 0;
  }

};
llvm::Optional<DcomputeAddrspacedType>
toDcomputeAddrspacedType(VarDeclaration *vd);
llvm::Optional<DcomputeAddrspacedType>
toDcomputeAddrspacedType(StructDeclaration *sd);
unsigned addressSpaceForVarDeclaration(VarDeclaration *sd);
#endif
