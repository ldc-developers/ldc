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

#include <utility>

class Dsymbol;
class Type;
class StructDeclaration;

bool isFromLDC_DComputeTypes(Dsymbol *sym);

std::pair<int, Type *> isDComputeTypesPointer(StructDeclaration *sd);
const std::pair<int, Type *> notDComputeTypesPointer =
    std::make_pair(-1, nullptr);

#endif
