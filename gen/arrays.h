//===-- gen/arrays.h - D array codegen helpers ------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Helper functions for manipulating D dynamic array (slice) types/values.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ARRAYS_H
#define LDC_GEN_ARRAYS_H

#include "lexer.h"
#include "gen/llvm.h"

class ArrayInitializer;
class ArrayLiteralExp;
class DSliceValue;
class DValue;
class Expression;
struct IRState;
struct Loc;
class Type;

llvm::StructType *DtoArrayType(Type *arrayTy);
llvm::StructType *DtoArrayType(LLType *elemTy);
llvm::ArrayType *DtoStaticArrayType(Type *sarrayTy);

/// Creates a (global) constant with the element data for the given arary
/// initializer. targetType is explicit because the frontend sometimes emits
/// ArrayInitializers for vectors typed as static arrays.
LLConstant *DtoConstArrayInitializer(ArrayInitializer *si, Type *targetType);

LLConstant *DtoConstSlice(LLConstant *dim, LLConstant *ptr, Type *type = 0);

/// Returns whether the array literal can be evaluated to a (LLVM) constant.
bool isConstLiteral(ArrayLiteralExp *ale);

/// Returns the constant for the given array literal expression.
llvm::Constant *arrayLiteralToConst(IRState *p, ArrayLiteralExp *ale);

/// Initializes a chunk of memory with the contents of an array literal.
///
/// dstMem is expected to be a pointer to the array allocation.
void initializeArrayLiteral(IRState *p, ArrayLiteralExp *ale, LLValue *dstMem);

void DtoArrayAssign(Loc &loc, DValue *lhs, DValue *rhs, int op,
                    bool canSkipPostblit);
void DtoSetArrayToNull(LLValue *v);

DSliceValue *DtoNewDynArray(Loc &loc, Type *arrayType, DValue *dim,
                            bool defaultInit = true);
DSliceValue *DtoNewMulDimDynArray(Loc &loc, Type *arrayType, DValue **dims,
                                  size_t ndims);
DSliceValue *DtoResizeDynArray(Loc &loc, Type *arrayType, DValue *array,
                               llvm::Value *newdim);

void DtoCatAssignElement(Loc &loc, Type *type, DValue *arr, Expression *exp);
DSliceValue *DtoCatAssignArray(Loc &loc, DValue *arr, Expression *exp);
DSliceValue *DtoCatArrays(Loc &loc, Type *type, Expression *e1, Expression *e2);
DSliceValue *DtoAppendDCharToString(Loc &loc, DValue *arr, Expression *exp);
DSliceValue *DtoAppendDCharToUnicodeString(Loc &loc, DValue *arr,
                                           Expression *exp);

LLValue *DtoArrayEquals(Loc &loc, TOK op, DValue *l, DValue *r);
LLValue *DtoArrayCompare(Loc &loc, TOK op, DValue *l, DValue *r);

LLValue *DtoDynArrayIs(TOK op, DValue *l, DValue *r);

LLValue *DtoArrayCastLength(Loc &loc, LLValue *len, LLType *elemty,
                            LLType *newelemty);

LLValue *DtoArrayLen(DValue *v);
LLValue *DtoArrayPtr(DValue *v);

DValue *DtoCastArray(Loc &loc, DValue *val, Type *to);

// generates an array bounds check
void DtoIndexBoundsCheck(Loc &loc, DValue *arr, DValue *index);

/// Inserts a call to the druntime function that throws the range error, with
/// the given location.
void DtoBoundsCheckFailCall(IRState *p, Loc &loc);

#endif // LDC_GEN_ARRAYS_H
