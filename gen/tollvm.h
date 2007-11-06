#ifndef LLVMDC_GEN_TOLLVM_H
#define LLVMDC_GEN_TOLLVM_H

// D -> LLVM helpers

struct DValue;

const llvm::Type* DtoType(Type* t);
bool DtoIsPassedByRef(Type* type);
Type* DtoDType(Type* t);

const llvm::FunctionType* DtoFunctionType(Type* t, const llvm::Type* thistype, bool ismain = false);
const llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl);
llvm::Function* DtoDeclareFunction(FuncDeclaration* fdecl);

const llvm::StructType* DtoDelegateType(Type* t);
llvm::Value* DtoNullDelegate(llvm::Value* v);
llvm::Value* DtoDelegateCopy(llvm::Value* dst, llvm::Value* src);
llvm::Value* DtoCompareDelegate(TOK op, llvm::Value* lhs, llvm::Value* rhs);

llvm::GlobalValue::LinkageTypes DtoLinkage(PROT prot, uint stc);
unsigned DtoCallingConv(LINK l);

llvm::Value* DtoPointedType(llvm::Value* ptr, llvm::Value* val);
llvm::Value* DtoBoolean(llvm::Value* val);

const llvm::Type* DtoSize_t();

void DtoMain();

void DtoCallClassDtors(TypeClass* tc, llvm::Value* instance);
void DtoInitClass(TypeClass* tc, llvm::Value* dst);

llvm::Constant* DtoConstInitializer(Type* type, Initializer* init);
elem* DtoInitializer(Initializer* init);

llvm::Function* LLVM_DeclareMemSet32();
llvm::Function* LLVM_DeclareMemSet64();
llvm::Function* LLVM_DeclareMemCpy32();
llvm::Function* LLVM_DeclareMemCpy64();

llvm::Value* DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* DtoGEPi(llvm::Value* ptr, unsigned i0, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* DtoGEPi(llvm::Value* ptr, unsigned i0, unsigned i1, const std::string& var, llvm::BasicBlock* bb=NULL);

llvm::Value* DtoRealloc(llvm::Value* ptr, const llvm::Type* ty);
llvm::Value* DtoRealloc(llvm::Value* ptr, llvm::Value* len);

void DtoAssert(llvm::Value* cond, llvm::Value* loc, llvm::Value* msg);

llvm::Value* DtoArgument(const llvm::Type* paramtype, Argument* fnarg, Expression* argexp);

llvm::Value* DtoNestedVariable(VarDeclaration* vd);

llvm::ConstantInt* DtoConstSize_t(size_t);
llvm::ConstantInt* DtoConstUint(unsigned i);
llvm::ConstantInt* DtoConstInt(int i);
llvm::Constant* DtoConstString(const char*);
llvm::Constant* DtoConstStringPtr(const char* str, const char* section = 0);
llvm::Constant* DtoConstBool(bool);

bool DtoIsTemplateInstance(Dsymbol* s);

void DtoLazyStaticInit(bool istempl, llvm::Value* gvar, Initializer* init, Type* t);

// llvm wrappers
void DtoMemCpy(llvm::Value* dst, llvm::Value* src, llvm::Value* nbytes);
bool DtoCanLoad(llvm::Value* ptr);
llvm::Value* DtoLoad(llvm::Value* src);
void DtoStore(llvm::Value* src, llvm::Value* dst);
llvm::Value* DtoBitCast(llvm::Value* v, const llvm::Type* t);

// basic operations
void DtoAssign(DValue* lhs, DValue* rhs);

// binary operations
DValue* DtoBinAdd(DValue* lhs, DValue* rhs);
DValue* DtoBinSub(DValue* lhs, DValue* rhs);
DValue* DtoBinMul(DValue* lhs, DValue* rhs);
DValue* DtoBinDiv(DValue* lhs, DValue* rhs);
DValue* DtoBinRem(DValue* lhs, DValue* rhs);

#include "enums.h"

#endif // LLVMDC_GEN_TOLLVM_H
