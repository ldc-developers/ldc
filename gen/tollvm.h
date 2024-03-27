//===-- gen/tollvm.h - General LLVM codegen helpers -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// General codegen helper constructs.
//
// TODO: Merge with gen/llvmhelpers.h, then refactor into sensible parts.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/tokens.h"
#include "dmd/mtype.h"
#include "gen/attributes.h"
#include "gen/llvm.h"
#include "gen/structs.h"

// D->LLVM type handling stuff

/* The function takes a d type and returns an appropriate llvm type.
 *
 * Notice that the function does not support function types with context
 * arguments.
 * DtoTypeFunction(FuncDeclaration*) is to be used instead.
 */
LLType *DtoType(Type *t);
// Uses DtoType(), but promotes i1 and void to i8.
LLType *DtoMemType(Type *t);
// Returns a pointer to the type returned by DtoMemType(t).
LLPointerType *DtoPtrToType(Type *t);

LLType *voidToI8(LLType *t);
LLType *i1ToI8(LLType *t);

// Removes all addrspace qualifications. float addrspace(1)** -> float**
// Use when comparing pointers LLType* for equality with `== ` when one side
// may be addrspace qualified.
LLType *stripAddrSpaces(LLType *v);

// Returns true if the type is a value type which LDC keeps exclusively in
// memory, referencing all values via LL pointers (structs and static arrays).
bool DtoIsInMemoryOnly(Type *type);

// Returns true if the callee uses sret (struct return).
// In that case, the caller needs to allocate the return value and pass its
// address as additional parameter to the callee, which will set it up.
bool DtoIsReturnInArg(CallExp *ce);

// Adds an appropriate attribute if the type should be zero or sign extended.
void DtoAddExtendAttr(Type *type, llvm::AttrBuilder &attrs);

// tuple helper
// takes a arguments list and makes a struct type out of them
// LLType* DtoStructTypeFromArguments(Arguments* arguments);

// delegate helpers
LLValue *DtoDelegateEquals(EXP op, LLValue *lhs, LLValue *rhs);

// Returns the LLVM linkage to use for the definition of the given symbol,
// based on whether it is a template or not.
typedef std::pair<llvm::GlobalValue::LinkageTypes, bool> LinkageWithCOMDAT;
LinkageWithCOMDAT DtoLinkage(Dsymbol *sym);

bool needsCOMDAT();
void setLinkage(LinkageWithCOMDAT lwc, llvm::GlobalObject *obj);
// Sets linkage and visibility of the specified IR symbol based on the specified
// D symbol.
void setLinkageAndVisibility(Dsymbol *sym, llvm::GlobalObject *obj);
// Hides or exports the specified IR symbol depending on its linkage,
// `-fvisibility` and the specified D symbol's visibility.
void setVisibility(Dsymbol *sym, llvm::GlobalObject *obj);

// some types
LLIntegerType *DtoSize_t();
LLStructType *DtoModuleReferenceType();

// getelementptr helpers
LLValue *DtoGEP1(LLType *pointeeTy, LLValue *ptr, LLValue *i0,
                 const char *name = "", llvm::BasicBlock *bb = nullptr);
LLValue *DtoGEP(LLType *pointeeTy, LLValue *ptr, LLValue *i0, LLValue *i1,
                const char *name = "", llvm::BasicBlock *bb = nullptr);

LLValue *DtoGEP1(LLType *pointeeTy, LLValue *ptr, unsigned i0,
                 const char *name = "", llvm::BasicBlock *bb = nullptr);
LLValue *DtoGEP(LLType *pointeeTy, LLValue *ptr, unsigned i0, unsigned i1,
                const char *name = "", llvm::BasicBlock *bb = nullptr);
LLConstant *DtoGEP(LLType *pointeeTy, LLConstant *ptr, unsigned i0,
                   unsigned i1);

LLValue *DtoGEP1i64(LLType *pointeeTy, LLValue *ptr, uint64_t i0,
                    const char *name = "", llvm::BasicBlock *bb = nullptr);

// to constant helpers
LLConstantInt *DtoConstSize_t(uint64_t);
LLConstantInt *DtoConstUlong(uint64_t i);
LLConstantInt *DtoConstLong(int64_t i);
LLConstantInt *DtoConstUint(unsigned i);
LLConstantInt *DtoConstInt(int i);
LLConstantInt *DtoConstUshort(uint16_t i);
LLConstantInt *DtoConstUbyte(unsigned char i);
LLConstant *DtoConstFP(Type *t, real_t value);

LLConstant *DtoConstCString(const char *);
LLConstant *DtoConstString(const char *);
LLConstant *DtoConstBool(bool);

// llvm wrappers
class DLValue;
LLValue *DtoLoad(DLValue *src, const char *name = "");
LLValue *DtoLoad(LLType *, LLValue *src, const char *name = "");
LLValue *DtoVolatileLoad(LLType *, LLValue *src, const char *name = "");
LLValue *DtoAlignedLoad(LLType *type, LLValue *src, const char *name = "");
void DtoStore(LLValue *src, LLValue *dst);
void DtoVolatileStore(LLValue *src, LLValue *dst);
void DtoStoreZextI8(LLValue *src, LLValue *dst);
void DtoAlignedStore(LLValue *src, LLValue *dst);
LLValue *DtoBitCast(LLValue *v, LLType *t, const llvm::Twine &name = "");
LLConstant *DtoBitCast(LLConstant *v, LLType *t);
LLValue *DtoInsertValue(LLValue *aggr, LLValue *v, unsigned idx,
                        const char *name = "");
LLValue *DtoExtractValue(LLValue *aggr, unsigned idx, const char *name = "");
LLValue *DtoInsertElement(LLValue *vec, LLValue *v, LLValue *idx,
                          const char *name = "");
LLValue *DtoExtractElement(LLValue *vec, LLValue *idx, const char *name = "");
LLValue *DtoInsertElement(LLValue *vec, LLValue *v, unsigned idx,
                          const char *name = "");
LLValue *DtoExtractElement(LLValue *vec, unsigned idx, const char *name = "");

// llvm::dyn_cast wrappers
LLPointerType *isaPointer(LLValue *v);
LLPointerType *isaPointer(LLType *t);
LLArrayType *isaArray(LLValue *v);
LLArrayType *isaArray(LLType *t);
LLStructType *isaStruct(LLValue *v);
LLStructType *isaStruct(LLType *t);
LLFunctionType *isaFunction(LLValue *v);
LLFunctionType *isaFunction(LLType *t);
LLConstant *isaConstant(LLValue *v);
LLConstantInt *isaConstantInt(LLValue *v);
llvm::Argument *isaArgument(LLValue *v);
LLGlobalVariable *isaGlobalVar(LLValue *v);

// llvm::T::get(...) wrappers
LLType *getI8Type();
LLPointerType *getPtrToType(LLType *t);
LLPointerType *getVoidPtrType();
LLPointerType *getVoidPtrType(llvm::LLVMContext &C);
llvm::ConstantPointerNull *getNullPtr(LLType *t);
LLConstant *getNullValue(LLType *t);

// type sizes
size_t getTypeBitSize(LLType *t);
size_t getTypeStoreSize(LLType *t);
size_t getTypeAllocSize(LLType *t);

// type alignments
unsigned int getABITypeAlign(LLType *t);

// pair type helpers
LLValue *DtoAggrPair(LLType *type, LLValue *V1, LLValue *V2,
                     const char *name = "");
LLValue *DtoAggrPair(LLValue *V1, LLValue *V2, const char *name = "");
LLValue *DtoSlicePaint(LLValue *aggr, LLType *as);

/**
 * Generates a call to llvm.memset.i32 (or i64 depending on architecture).
 * @param dst Destination memory.
 * @param val The value to set.
 * @param nbytes Number of bytes to overwrite.
 * @param align The minimum alignment of the destination memory.
 */
void DtoMemSet(LLValue *dst, LLValue *val, LLValue *nbytes, unsigned align = 1);

/**
 * Generates a call to llvm.memset.i32 (or i64 depending on architecture).
 * @param dst Destination memory.
 * @param nbytes Number of bytes to overwrite.
 * @param align The minimum alignment of the destination memory.
 */
void DtoMemSetZero(LLType *type, LLValue *dst, LLValue *nbytes, unsigned align = 1);

/**
 * The same as DtoMemSetZero but figures out the size itself based on the
 * dst pointee.
 * @param dst Destination memory.
 * @param align The minimum alignment of the destination memory.
 */
void DtoMemSetZero(LLType *type, LLValue *dst, unsigned align = 1);

/**
 * Generates a call to llvm.memcpy.i32 (or i64 depending on architecture).
 * @param dst Destination memory.
 * @param src Source memory.
 * @param nbytes Number of bytes to copy.
 * @param align The minimum alignment of the source and destination memory.
 */
void DtoMemCpy(LLValue *dst, LLValue *src, LLValue *nbytes, unsigned align = 1);

/**
 * The same as DtoMemCpy but figures out the size itself based on the dst
 * pointee.
 * @param dst Destination memory.
 * @param src Source memory.
 * @param withPadding Use the dst pointee's padded size, not its store size.
 * @param align The minimum alignment of the source and destination memory.
 */
void DtoMemCpy(LLType *type, LLValue *dst, LLValue *src, bool withPadding = false,
               unsigned align = 1);

/**
 * Generates a call to C memcmp.
 */
LLValue *DtoMemCmp(LLValue *lhs, LLValue *rhs, LLValue *nbytes);
