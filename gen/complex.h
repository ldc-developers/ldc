//===-- gen/cl_helpers.h - Complex number code generation -------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for generating code for D complex number operations.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_COMPLEX_H
#define LDC_GEN_COMPLEX_H

#include "dmd/tokens.h"
#include "gen/dvalue.h"

struct Loc;
class Type;
namespace llvm {
class Constant;
class StructType;
class Type;
class Value;
}

llvm::StructType *DtoComplexType(Type *t);
llvm::Type *DtoComplexBaseType(Type *t);

llvm::Constant *DtoConstComplex(Type *t, real_t re, real_t im);

llvm::Constant *DtoComplexShuffleMask(unsigned a, unsigned b);

DValue *DtoComplex(Loc &loc, Type *to, DValue *val);

void DtoComplexSet(llvm::Value *c, llvm::Value *re, llvm::Value *im);

void DtoGetComplexParts(Loc &loc, Type *to, DValue *c, DValue *&re,
                        DValue *&im);
void DtoGetComplexParts(Loc &loc, Type *to, DValue *c, llvm::Value *&re,
                        llvm::Value *&im);

DImValue *DtoComplexAdd(Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexMin(Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexMul(Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexDiv(Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexMod(Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexNeg(Loc &loc, Type *type, DRValue *val);

llvm::Value *DtoComplexEquals(Loc &loc, TOK op, DValue *lhs, DValue *rhs);

DValue *DtoCastComplex(Loc &loc, DValue *val, Type *to);

#endif // LDC_GEN_COMPLEX_H
