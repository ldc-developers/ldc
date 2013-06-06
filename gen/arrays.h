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

struct ArrayInitializer;
class DSliceValue;
class DValue;
struct Expression;
struct Loc;
struct Type;

llvm::StructType* DtoArrayType(Type* arrayTy);
llvm::StructType* DtoArrayType(LLType* elemTy);
llvm::ArrayType* DtoStaticArrayType(Type* sarrayTy);

LLType* DtoConstArrayInitializerType(ArrayInitializer* arrinit);
LLConstant* DtoConstArrayInitializer(ArrayInitializer* si);
LLConstant* DtoConstSlice(LLConstant* dim, LLConstant* ptr, Type *type = 0);

void DtoArrayCopySlices(DSliceValue* dst, DSliceValue* src);
void DtoArrayCopyToSlice(DSliceValue* dst, DValue* src);

void DtoArrayInit(Loc& loc, DValue* array, DValue* value, int op);
Type *DtoArrayElementType(Type *arrayType);
bool arrayNeedsPostblit(Type *t);
void DtoArrayAssign(DValue *from, DValue *to, int op);
void DtoArraySetAssign(Loc &loc, DValue *array, DValue *value, int op);
void DtoSetArray(DValue* array, LLValue* dim, LLValue* ptr);
void DtoSetArrayToNull(LLValue* v);

DSliceValue* DtoNewDynArray(Loc& loc, Type* arrayType, DValue* dim, bool defaultInit=true);
DSliceValue* DtoNewMulDimDynArray(Loc& loc, Type* arrayType, DValue** dims, size_t ndims, bool defaultInit=true);
DSliceValue* DtoResizeDynArray(Type* arrayType, DValue* array, llvm::Value* newdim);

void DtoCatAssignElement(Loc& loc, Type* type, DValue* arr, Expression* exp);
DSliceValue* DtoCatAssignArray(DValue* arr, Expression* exp);
DSliceValue* DtoCatArrays(Type* type, Expression* e1, Expression* e2);
DSliceValue* DtoAppendDCharToString(DValue* arr, Expression* exp);
DSliceValue* DtoAppendDCharToUnicodeString(DValue* arr, Expression* exp);

void DtoStaticArrayCopy(LLValue* dst, LLValue* src);

LLValue* DtoArrayEquals(Loc& loc, TOK op, DValue* l, DValue* r);
LLValue* DtoArrayCompare(Loc& loc, TOK op, DValue* l, DValue* r);

LLValue* DtoDynArrayIs(TOK op, DValue* l, DValue* r);

LLValue* DtoArrayCastLength(LLValue* len, LLType* elemty, LLType* newelemty);

LLValue* DtoArrayLen(DValue* v);
LLValue* DtoArrayPtr(DValue* v);

DValue* DtoCastArray(Loc& loc, DValue* val, Type* to);

// generates an array bounds check
void DtoArrayBoundsCheck(Loc& loc, DValue* arr, DValue* index, DValue* lowerBound = 0);

#endif // LDC_GEN_ARRAYS_H
