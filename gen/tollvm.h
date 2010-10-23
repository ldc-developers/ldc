#ifndef LDC_GEN_TOLLVM_H
#define LDC_GEN_TOLLVM_H

#include "gen/llvm.h"
#include "lexer.h"
#include "mtype.h"
#include "attrib.h"
#include "declaration.h"

#include "gen/structs.h"

// D->LLVM type handling stuff
const LLType* DtoType(Type* t);

// same as DtoType except it converts 'void' to 'i8'
const LLType* DtoTypeNotVoid(Type* t);

// returns true is the type must be passed by pointer
bool DtoIsPassedByRef(Type* type);

// should argument be zero or sign extended
unsigned DtoShouldExtend(Type* type);

// tuple helper
// takes a arguments list and makes a struct type out of them
//const LLType* DtoStructTypeFromArguments(Arguments* arguments);

// delegate helpers
LLValue* DtoDelegateEquals(TOK op, LLValue* lhs, LLValue* rhs);

// return linkage type for symbol using the current ir state for context
LLGlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym);
LLGlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym);
LLGlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym);

// TODO: this one should be removed!!!
LLValue* DtoPointedType(LLValue* ptr, LLValue* val);

// some types
const LLIntegerType* DtoSize_t();
const LLStructType* DtoInterfaceInfoType();
const LLStructType* DtoMutexType();
const LLStructType* DtoModuleReferenceType();

// getelementptr helpers
LLValue* DtoGEP1(LLValue* ptr, LLValue* i0, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var=NULL, llvm::BasicBlock* bb=NULL);

LLValue* DtoGEPi1(LLValue* ptr, unsigned i0, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLConstant* DtoGEPi(LLConstant* ptr, unsigned i0, unsigned i1);

// to constant helpers
LLConstantInt* DtoConstSize_t(uint64_t);
LLConstantInt* DtoConstUint(unsigned i);
LLConstantInt* DtoConstInt(int i);
LLConstantInt* DtoConstUbyte(unsigned char i);
LLConstant* DtoConstFP(Type* t, long double value);

LLConstant* DtoConstString(const char*);
LLConstant* DtoConstStringPtr(const char* str, const char* section = 0);
LLConstant* DtoConstBool(bool);

// llvm wrappers
LLValue* DtoLoad(LLValue* src, const char* name=0);
LLValue* DtoAlignedLoad(LLValue* src, const char* name=0);
void DtoStore(LLValue* src, LLValue* dst);
void DtoAlignedStore(LLValue* src, LLValue* dst);
LLValue* DtoBitCast(LLValue* v, const LLType* t, const char* name=0);
LLConstant* DtoBitCast(LLConstant* v, const LLType* t);
LLValue* DtoInsertValue(LLValue* aggr, LLValue* v, unsigned idx);
LLValue* DtoExtractValue(LLValue* aggr, unsigned idx);

// llvm::dyn_cast wrappers
const LLPointerType* isaPointer(LLValue* v);
const LLPointerType* isaPointer(const LLType* t);
const LLArrayType* isaArray(LLValue* v);
const LLArrayType* isaArray(const LLType* t);
const LLStructType* isaStruct(LLValue* v);
const LLStructType* isaStruct(const LLType* t);
const LLFunctionType* isaFunction(LLValue* v);
const LLFunctionType* isaFunction(const LLType* t);
LLConstant* isaConstant(LLValue* v);
LLConstantInt* isaConstantInt(LLValue* v);
llvm::Argument* isaArgument(LLValue* v);
LLGlobalVariable* isaGlobalVar(LLValue* v);

// llvm::T::get(...) wrappers
const LLPointerType* getPtrToType(const LLType* t);
const LLPointerType* getVoidPtrType();
llvm::ConstantPointerNull* getNullPtr(const LLType* t);
LLConstant* getNullValue(const LLType* t);

// type sizes
size_t getTypeBitSize(const LLType* t);
size_t getTypeStoreSize(const LLType* t);
size_t getTypePaddedSize(const LLType* t);

// type alignments
unsigned char getABITypeAlign(const LLType* t);
unsigned char getPrefTypeAlign(const LLType* t);

// get biggest type, for unions ...
const LLType* getBiggestType(const LLType** begin, size_t n);

// pair type helpers
LLValue* DtoAggrPair(const LLType* type, LLValue* V1, LLValue* V2, const char* name = 0);
LLValue* DtoAggrPair(LLValue* V1, LLValue* V2, const char* name = 0);
LLValue* DtoAggrPaint(LLValue* aggr, const LLType* as);
LLValue* DtoAggrPairSwap(LLValue* aggr);

/**
 * Generates a call to llvm.memset.i32 (or i64 depending on architecture).
 * @param dst Destination memory.
 * @param val The value to set.
 * @param nbytes Number of bytes to overwrite.
 */
void DtoMemSet(LLValue* dst, LLValue* val, LLValue* nbytes);

/**
 * Generates a call to llvm.memset.i32 (or i64 depending on architecture).
 * @param dst Destination memory.
 * @param nbytes Number of bytes to overwrite.
 */
void DtoMemSetZero(LLValue* dst, LLValue* nbytes);

/**
 * Generates a call to llvm.memcpy.i32 (or i64 depending on architecture).
 * @param dst Destination memory.
 * @param src Source memory.
 * @param nbytes Number of bytes to copy.
 * @param align The minimum alignment of the source and destination memory.
 */
void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes, unsigned align = 1);

/**
 * Generates a call to C memcmp.
 */
LLValue* DtoMemCmp(LLValue* lhs, LLValue* rhs, LLValue* nbytes);

/**
 * The same as DtoMemSetZero but figures out the size itself by "dereferencing" the v pointer once.
 * @param v Destination memory.
 */
void DtoAggrZeroInit(LLValue* v);

/**
 * The same as DtoMemCpy but figures out the size itself by "dereferencing" dst the pointer once.
 * @param dst Destination memory.
 * @param src Source memory.
 */
void DtoAggrCopy(LLValue* dst, LLValue* src);

/**
 * Generates a call to llvm.memory.barrier
 * @param ll load-load
 * @param ls load-store
 * @param sl store-load
 * @param ss store-store
 * @param device special device flag
 */
void DtoMemoryBarrier(bool ll, bool ls, bool sl, bool ss, bool device=false);

#include "enums.h"

#endif // LDC_GEN_TOLLVM_H
