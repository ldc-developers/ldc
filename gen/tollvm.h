#ifndef LLVMDC_GEN_TOLLVM_H
#define LLVMDC_GEN_TOLLVM_H

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

// returns if the type should be returned in a hidden pointer arguement
bool DtoIsReturnedInArg(Type* type);

// resolve typedefs to their real type.
// TODO should probably be removed in favor of DMD's Type::toBasetype
Type* DtoDType(Type* t);

// delegate helpers
const LLStructType* DtoDelegateType(Type* t);
LLValue* DtoDelegateEquals(TOK op, LLValue* lhs, LLValue* rhs);

// return linkage type for symbol using the current ir state for context
LLGlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym);
LLGlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym);
LLGlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym);

// convert DMD calling conv to LLVM
unsigned DtoCallingConv(LINK l);

// TODO: this one should be removed!!!
LLValue* DtoPointedType(LLValue* ptr, LLValue* val);

// casts any value to a boolean
LLValue* DtoBoolean(DValue* dval);

// some types
const LLType* DtoSize_t();
const LLStructType* DtoInterfaceInfoType();
const LLStructType* DtoMutexType();
const LLStructType* DtoModuleReferenceType();

// getelementptr helpers
LLValue* DtoGEP1(LLValue* ptr, LLValue* i0, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi(LLValue* ptr, const DStructIndexVector& src, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi1(LLValue* ptr, unsigned i0, const char* var=NULL, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var=NULL, llvm::BasicBlock* bb=NULL);

// to constant helpers
LLConstantInt* DtoConstSize_t(size_t);
LLConstantInt* DtoConstUint(unsigned i);
LLConstantInt* DtoConstInt(int i);
LLConstantInt* DtoConstUbyte(unsigned char i);
llvm::ConstantFP* DtoConstFP(Type* t, long double value);

LLConstant* DtoConstString(const char*);
LLConstant* DtoConstStringPtr(const char* str, const char* section = 0);
LLConstant* DtoConstBool(bool);

// llvm wrappers
bool DtoCanLoad(LLValue* ptr);
LLValue* DtoLoad(LLValue* src, const char* name=0);
void DtoStore(LLValue* src, LLValue* dst);
LLValue* DtoBitCast(LLValue* v, const LLType* t, const char* name=0);

// llvm::dyn_cast wrappers
const LLPointerType* isaPointer(LLValue* v);
const LLPointerType* isaPointer(const LLType* t);
const LLArrayType* isaArray(LLValue* v);
const LLArrayType* isaArray(const LLType* t);
const LLStructType* isaStruct(LLValue* v);
const LLStructType* isaStruct(const LLType* t);
LLConstant* isaConstant(LLValue* v);
LLConstantInt* isaConstantInt(LLValue* v);
llvm::Argument* isaArgument(LLValue* v);
LLGlobalVariable* isaGlobalVar(LLValue* v);

// llvm::T::get(...) wrappers
const LLPointerType* getPtrToType(const LLType* t);
const LLPointerType* getVoidPtrType();
llvm::ConstantPointerNull* getNullPtr(const LLType* t);

// type sizes
size_t getTypeBitSize(const LLType* t);
size_t getTypeStoreSize(const LLType* t);
size_t getABITypeSize(const LLType* t);
// type alignments
unsigned char getABITypeAlign(const LLType* t);
unsigned char getPrefTypeAlign(const LLType* t);

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
 */
void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes);

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

#endif // LLVMDC_GEN_TOLLVM_H
