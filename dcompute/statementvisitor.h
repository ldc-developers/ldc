//===-- dcompute/statementvisitor.h -----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DCOMPUTE_STATEMENTVISITOR_H
#define LDC_DCOMPUTE_STATEMENTVISITOR_H
class Visitor;
struct IRState;
Visitor* createDCopmuteToIRVisitor(IRState *irs);
#endif 
