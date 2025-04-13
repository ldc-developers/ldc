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

#pragma once

#include "dmd/tokens.h"
#include "gen/llvm.h"

class ArrayInitializer;
class ArrayLiteralExp;
class DSliceValue;
class DValue;
class Expression;
struct IRState;
struct Loc;
class Type;

llvm::ArrayType *DtoStaticArrayType(Type *sarrayTy);

/// Creates a (global) constant with the element data for the given arary
/// initializer. targetType is explicit because the frontend sometimes emits
/// ArrayInitializers for vectors typed as static arrays.
LLConstant *DtoConstArrayInitializer(ArrayInitializer *si, Type *targetType,
                                     const bool isCfile);

LLConstant *DtoConstSlice(LLConstant *dim, LLConstant *ptr);

/// Returns the element at position idx of the literal (assumed to be in range).
Expression *indexArrayLiteral(ArrayLiteralExp *ale, unsigned idx);

/// Returns whether the array literal can be evaluated to a (LLVM) constant.
/// immutableType indicates whether the literal is used to initialize an
/// immutable type, in which case allocated dynamic arrays are considered
/// constant too.
bool isConstLiteral(Expression *e, bool immutableType = false);

/// Returns the constant for the given array literal expression.
llvm::Constant *arrayLiteralToConst(IRState *p, ArrayLiteralExp *ale);

/// Initializes a chunk of memory with the contents of an array literal.
///
/// dstMem is expected to be a pointer to the array allocation.
void initializeArrayLiteral(IRState *p, ArrayLiteralExp *ale,
                            LLValue *dstMem, LLType *dstType);

void DtoArrayAssign(Loc loc, DValue *lhs, DValue *rhs, EXP op,
                    bool canSkipPostblit);
void DtoSetArrayToNull(DValue *v);

DSliceValue *DtoNewDynArray(Loc loc, Type *arrayType, DValue *dim,
                            bool defaultInit = true);

DSliceValue *DtoCatArrays(Loc loc, Type *type, Expression *e1,
                          Expression *e2);
DSliceValue *DtoAppendDCharToString(Loc loc, DValue *arr, Expression *exp);
DSliceValue *DtoAppendDCharToUnicodeString(Loc loc, DValue *arr,
                                           Expression *exp);

LLValue *DtoArrayEquals(Loc loc, EXP op, DValue *l, DValue *r);

LLValue *DtoDynArrayIs(EXP op, DValue *l, DValue *r);

LLValue *DtoArrayLen(DValue *v);
LLValue *DtoArrayPtr(DValue *v);

DValue *DtoCastArray(Loc loc, DValue *val, Type *to);

// generates an array bounds check
void DtoIndexBoundsCheck(Loc loc, DValue *arr, DValue *index);

/// Inserts a call to the druntime function that throws the range error, with
/// the given location.
void emitRangeError(IRState *irs, Loc loc);
void emitArraySliceError(IRState *irs, Loc loc, LLValue *lower,
                         LLValue *upper, LLValue *length);
void emitArrayIndexError(IRState *irs, Loc loc, LLValue *index,
                         LLValue *length);
