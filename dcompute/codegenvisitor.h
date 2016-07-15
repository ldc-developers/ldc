//===-- dcompute/codegenvisitor.h - LDC -------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#ifndef LDC_DCOMPUTE_CODEGENVISTOR_H
#define LDC_DCOMPUTE_CODEGENVISTOR_H
#include "Visitor.h"
#include "gen/irstate.h"
#include "dcompute/target.h"
void DcomputeDeclaration_codegen(Dsymbol *decl, IRState *irs, DComputeTarget &dct);
#endif 
