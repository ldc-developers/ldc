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

#pragma once

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

DValue *DtoComplex(const Loc &loc, Type *to, DValue *val);

void DtoComplexSet(llvm::Value *c, llvm::Value *re, llvm::Value *im);

void DtoGetComplexParts(const Loc &loc, Type *to, DValue *c, DValue *&re,
                        DValue *&im);
void DtoGetComplexParts(const Loc &loc, Type *to, DValue *c, llvm::Value *&re,
                        llvm::Value *&im);

DImValue *DtoComplexAdd(const Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexMin(const Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexMul(const Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexDiv(const Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexMod(const Loc &loc, Type *type, DRValue *lhs, DRValue *rhs);
DImValue *DtoComplexNeg(const Loc &loc, Type *type, DRValue *val);

llvm::Value *DtoComplexEquals(const Loc &loc, EXP op, DValue *lhs, DValue *rhs);

DValue *DtoCastComplex(const Loc &loc, DValue *val, Type *to);
