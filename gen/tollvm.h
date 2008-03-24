#ifndef LLVMDC_GEN_TOLLVM_H
#define LLVMDC_GEN_TOLLVM_H

#include "gen/llvm.h"
#include "lexer.h"
#include "mtype.h"
#include "attrib.h"
#include "declaration.h"

// D->LLVM type handling stuff
const llvm::Type* DtoType(Type* t);
bool DtoIsPassedByRef(Type* type);

// resolve typedefs to their real type.
// TODO should probably be removed in favor of DMD's Type::toBasetype
Type* DtoDType(Type* t);

// delegate helpers
const llvm::StructType* DtoDelegateType(Type* t);
llvm::Value* DtoNullDelegate(llvm::Value* v);
llvm::Value* DtoDelegateCopy(llvm::Value* dst, llvm::Value* src);
llvm::Value* DtoCompareDelegate(TOK op, llvm::Value* lhs, llvm::Value* rhs);

// return linkage type for symbol using the current ir state for context
llvm::GlobalValue::LinkageTypes DtoLinkage(Dsymbol* sym);
llvm::GlobalValue::LinkageTypes DtoInternalLinkage(Dsymbol* sym);
llvm::GlobalValue::LinkageTypes DtoExternalLinkage(Dsymbol* sym);

// convert DMD calling conv to LLVM
unsigned DtoCallingConv(LINK l);

// TODO: this one should be removed!!!
llvm::Value* DtoPointedType(llvm::Value* ptr, llvm::Value* val);

// casts any value to a boolean
llvm::Value* DtoBoolean(llvm::Value* val);

// some types
const llvm::Type* DtoSize_t();
const llvm::StructType* DtoInterfaceInfoType();

// initializer helpers
llvm::Constant* DtoConstInitializer(Type* type, Initializer* init);
llvm::Constant* DtoConstFieldInitializer(Type* type, Initializer* init);
DValue* DtoInitializer(Initializer* init);

// declaration of memset/cpy intrinsics
llvm::Function* LLVM_DeclareMemSet32();
llvm::Function* LLVM_DeclareMemSet64();
llvm::Function* LLVM_DeclareMemCpy32();
llvm::Function* LLVM_DeclareMemCpy64();

// getelementptr helpers
llvm::Value* DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* DtoGEPi(llvm::Value* ptr, unsigned i0, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* DtoGEPi(llvm::Value* ptr, unsigned i0, unsigned i1, const std::string& var, llvm::BasicBlock* bb=NULL);

// dynamic memory helpers
llvm::Value* DtoRealloc(llvm::Value* ptr, const llvm::Type* ty);
llvm::Value* DtoRealloc(llvm::Value* ptr, llvm::Value* len);

// assertion generator
void DtoAssert(Loc* loc, DValue* msg);

// nested variable/class helpers
llvm::Value* DtoNestedContext(FuncDeclaration* func);
llvm::Value* DtoNestedVariable(VarDeclaration* vd);

// annotation generator
void DtoAnnotation(const char* str);

// to constant helpers
llvm::ConstantInt* DtoConstSize_t(size_t);
llvm::ConstantInt* DtoConstUint(unsigned i);
llvm::ConstantInt* DtoConstInt(int i);
llvm::ConstantFP* DtoConstFP(Type* t, long double value);

llvm::Constant* DtoConstString(const char*);
llvm::Constant* DtoConstStringPtr(const char* str, const char* section = 0);
llvm::Constant* DtoConstBool(bool);
llvm::Constant* DtoConstNullPtr(const llvm::Type* t);

// is template instance check
bool DtoIsTemplateInstance(Dsymbol* s);

// generates lazy static initialization code for a global variable
void DtoLazyStaticInit(bool istempl, llvm::Value* gvar, Initializer* init, Type* t);

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
void DtoMemSetZero(llvm::Value* dst, llvm::Value* nbytes);
void DtoMemCpy(llvm::Value* dst, llvm::Value* src, llvm::Value* nbytes);
bool DtoCanLoad(llvm::Value* ptr);
llvm::Value* DtoLoad(llvm::Value* src);
void DtoStore(llvm::Value* src, llvm::Value* dst);
llvm::Value* DtoBitCast(llvm::Value* v, const llvm::Type* t, const char* name=0);

// llvm::dyn_cast wrappers
const llvm::PointerType* isaPointer(llvm::Value* v);
const llvm::PointerType* isaPointer(const llvm::Type* t);
const llvm::ArrayType* isaArray(llvm::Value* v);
const llvm::ArrayType* isaArray(const llvm::Type* t);
const llvm::StructType* isaStruct(llvm::Value* v);
const llvm::StructType* isaStruct(const llvm::Type* t);
llvm::Constant* isaConstant(llvm::Value* v);
llvm::ConstantInt* isaConstantInt(llvm::Value* v);
llvm::Argument* isaArgument(llvm::Value* v);
llvm::GlobalVariable* isaGlobalVar(llvm::Value* v);

// llvm::T::get(...) wrappers
const llvm::PointerType* getPtrToType(const llvm::Type* t);
llvm::ConstantPointerNull* getNullPtr(const llvm::Type* t);

// type sizes
size_t getTypeBitSize(const llvm::Type* t);
size_t getTypeStoreSize(const llvm::Type* t);
size_t getABITypeSize(const llvm::Type* t);

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
