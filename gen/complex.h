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

#include "lexer.h"
#include "longdouble.h"

class DValue;
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

llvm::Constant *DtoConstComplex(Type *t, longdouble re, longdouble im);

llvm::Constant *DtoComplexShuffleMask(unsigned a, unsigned b);

DValue *DtoComplex(Loc &loc, Type *to, DValue *val);

void DtoComplexSet(llvm::Value *c, llvm::Value *re, llvm::Value *im);

void DtoGetComplexParts(Loc &loc, Type *to, DValue *c, DValue *&re,
                        DValue *&im);
void DtoGetComplexParts(Loc &loc, Type *to, DValue *c, llvm::Value *&re,
                        llvm::Value *&im);

DValue *DtoComplexAdd(Loc &loc, Type *type, DValue *lhs, DValue *rhs);
DValue *DtoComplexSub(Loc &loc, Type *type, DValue *lhs, DValue *rhs);
DValue *DtoComplexMul(Loc &loc, Type *type, DValue *lhs, DValue *rhs);
DValue *DtoComplexDiv(Loc &loc, Type *type, DValue *lhs, DValue *rhs);
DValue *DtoComplexRem(Loc &loc, Type *type, DValue *lhs, DValue *rhs);
DValue *DtoComplexNeg(Loc &loc, Type *type, DValue *val);

llvm::Value *DtoComplexEquals(Loc &loc, TOK op, DValue *lhs, DValue *rhs);

DValue *DtoCastComplex(Loc &loc, DValue *val, Type *to);

#endif // LDC_GEN_COMPLEX_H
