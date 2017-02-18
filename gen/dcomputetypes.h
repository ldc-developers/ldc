//===-- gen/dcomputetypes.h -------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DCOMPUTETYPES_H
#define LDC_GEN_DCOMPUTETYPES_H

#include "ddmd/aggregate.h"
#include "ddmd/mtype.h"

class Dsymbol;
class Type;


bool isFromLDC_DComputeTypes(Dsymbol *sym);

struct DcomputePointer {
    int addrspace;
    Type* type;
    DcomputePointer(StructDeclaration *sd);
    operator bool() const { return addrspace != -1 && type; }
};

#endif
