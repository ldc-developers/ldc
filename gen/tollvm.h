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
bool DtoIsPassedByRef(Type* type);
bool DtoIsReturnedInArg(Type* type);

// resolve typedefs to their real type.
// TODO should probably be removed in favor of DMD's Type::toBasetype
Type* DtoDType(Type* t);

// delegate helpers
const llvm::StructType* DtoDelegateType(Type* t);
void DtoDelegateToNull(LLValue* v);
void DtoDelegateCopy(LLValue* dst, LLValue* src);
LLValue* DtoDelegateCompare(TOK op, LLValue* lhs, LLValue* rhs);

// return linkage type for symbol using the current ir state for context
llvm::GlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym);
llvm::GlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym);
llvm::GlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym);

// convert DMD calling conv to LLVM
unsigned DtoCallingConv(LINK l);

// TODO: this one should be removed!!!
LLValue* DtoPointedType(LLValue* ptr, LLValue* val);

// casts any value to a boolean
LLValue* DtoBoolean(LLValue* val);

// some types
const LLType* DtoSize_t();
const llvm::StructType* DtoInterfaceInfoType();

// getting typeinfo of type, base=true casts to object.TypeInfo
LLConstant* DtoTypeInfoOf(Type* ty, bool base=true);

// initializer helpers
LLConstant* DtoConstInitializer(Type* type, Initializer* init);
LLConstant* DtoConstFieldInitializer(Type* type, Initializer* init);
DValue* DtoInitializer(Initializer* init);

// declaration of memset/cpy intrinsics
llvm::Function* LLVM_DeclareMemSet32();
llvm::Function* LLVM_DeclareMemSet64();
llvm::Function* LLVM_DeclareMemCpy32();
llvm::Function* LLVM_DeclareMemCpy64();

// getelementptr helpers
LLValue* DtoGEP(LLValue* ptr, LLValue* i0, LLValue* i1, const char* var, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi(LLValue* ptr, const DStructIndexVector& src, const char* var, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi(LLValue* ptr, unsigned i0, const char* var, llvm::BasicBlock* bb=NULL);
LLValue* DtoGEPi(LLValue* ptr, unsigned i0, unsigned i1, const char* var, llvm::BasicBlock* bb=NULL);

// dynamic memory helpers
LLValue* DtoNew(Type* newtype);
void DtoDeleteMemory(LLValue* ptr);
void DtoDeleteClass(LLValue* inst);
void DtoDeleteInterface(LLValue* inst);
void DtoDeleteArray(DValue* arr);

// assertion generator
void DtoAssert(Loc* loc, DValue* msg);

// nested variable/class helpers
LLValue* DtoNestedContext(FuncDeclaration* func);
LLValue* DtoNestedVariable(VarDeclaration* vd);

// annotation generator
void DtoAnnotation(const char* str);

// to constant helpers
llvm::ConstantInt* DtoConstSize_t(size_t);
llvm::ConstantInt* DtoConstUint(unsigned i);
llvm::ConstantInt* DtoConstInt(int i);
llvm::ConstantFP* DtoConstFP(Type* t, long double value);

LLConstant* DtoConstString(const char*);
LLConstant* DtoConstStringPtr(const char* str, const char* section = 0);
LLConstant* DtoConstBool(bool);

// is template instance check
bool DtoIsTemplateInstance(Dsymbol* s);

// generates lazy static initialization code for a global variable
void DtoLazyStaticInit(bool istempl, LLValue* gvar, Initializer* init, Type* t);

// these are all basically drivers for the codegeneration called by the main loop
void DtoResolveDsymbol(Dsymbol* dsym);
void DtoDeclareDsymbol(Dsymbol* dsym);
void DtoDefineDsymbol(Dsymbol* dsym);
void DtoConstInitDsymbol(Dsymbol* dsym);
void DtoConstInitGlobal(VarDeclaration* vd);
void DtoEmptyResolveList();
void DtoEmptyDeclareList();
void DtoEmptyConstInitList();
void DtoEmptyAllLists();
void DtoForceDeclareDsymbol(Dsymbol* dsym);
void DtoForceConstInitDsymbol(Dsymbol* dsym);
void DtoForceDefineDsymbol(Dsymbol* dsym);

// llvm wrappers
void DtoMemSetZero(LLValue* dst, LLValue* nbytes);
void DtoMemCpy(LLValue* dst, LLValue* src, LLValue* nbytes);
void DtoMemoryBarrier(bool ll, bool ls, bool sl, bool ss, bool device=false);
bool DtoCanLoad(LLValue* ptr);
LLValue* DtoLoad(LLValue* src, const char* name=0);
void DtoStore(LLValue* src, LLValue* dst);
LLValue* DtoBitCast(LLValue* v, const LLType* t, const char* name=0);

// llvm::dyn_cast wrappers
const llvm::PointerType* isaPointer(LLValue* v);
const llvm::PointerType* isaPointer(const LLType* t);
const llvm::ArrayType* isaArray(LLValue* v);
const llvm::ArrayType* isaArray(const LLType* t);
const llvm::StructType* isaStruct(LLValue* v);
const llvm::StructType* isaStruct(const LLType* t);
LLConstant* isaConstant(LLValue* v);
llvm::ConstantInt* isaConstantInt(LLValue* v);
llvm::Argument* isaArgument(LLValue* v);
llvm::GlobalVariable* isaGlobalVar(LLValue* v);

// llvm::T::get(...) wrappers
const llvm::PointerType* getPtrToType(const LLType* t);
const llvm::PointerType* getVoidPtrType();
llvm::ConstantPointerNull* getNullPtr(const LLType* t);

// type sizes
size_t getTypeBitSize(const LLType* t);
size_t getTypeStoreSize(const LLType* t);
size_t getABITypeSize(const LLType* t);

// basic operations
void DtoAssign(DValue* lhs, DValue* rhs);

// casts
DValue* DtoCastInt(DValue* val, Type* to);
DValue* DtoCastPtr(DValue* val, Type* to);
DValue* DtoCastFloat(DValue* val, Type* to);
DValue* DtoCastComplex(DValue* val, Type* to);
DValue* DtoCast(DValue* val, Type* to);

// binary operations
DValue* DtoBinAdd(DValue* lhs, DValue* rhs);
DValue* DtoBinSub(DValue* lhs, DValue* rhs);
DValue* DtoBinMul(DValue* lhs, DValue* rhs);
DValue* DtoBinDiv(DValue* lhs, DValue* rhs);
DValue* DtoBinRem(DValue* lhs, DValue* rhs);

#include "enums.h"

#endif // LLVMDC_GEN_TOLLVM_H
