// D -> LLVM helpers

struct StructInitializer;

const llvm::Type* LLVM_DtoType(Type* t);
bool LLVM_DtoIsPassedByRef(Type* type);
Type* LLVM_DtoDType(Type* t);

const llvm::Type* LLVM_DtoStructType(Type* t);
llvm::Value* LLVM_DtoStructZeroInit(llvm::Value* v);
llvm::Value* LLVM_DtoStructCopy(llvm::Value* dst, llvm::Value* src);
llvm::Constant* LLVM_DtoConstStructInitializer(StructInitializer* si);

const llvm::FunctionType* LLVM_DtoFunctionType(Type* t, const llvm::Type* thistype, bool ismain = false);
const llvm::FunctionType* LLVM_DtoFunctionType(FuncDeclaration* fdecl);
llvm::Function* LLVM_DtoDeclareFunction(FuncDeclaration* fdecl);

const llvm::StructType* LLVM_DtoDelegateType(Type* t);
llvm::Value* LLVM_DtoNullDelegate(llvm::Value* v);
llvm::Value* LLVM_DtoDelegateCopy(llvm::Value* dst, llvm::Value* src);
llvm::Value* LLVM_DtoCompareDelegate(TOK op, llvm::Value* lhs, llvm::Value* rhs);

llvm::GlobalValue::LinkageTypes LLVM_DtoLinkage(PROT prot, uint stc);
unsigned LLVM_DtoCallingConv(LINK l);

llvm::Value* LLVM_DtoPointedType(llvm::Value* ptr, llvm::Value* val);
llvm::Value* LLVM_DtoBoolean(llvm::Value* val);

const llvm::Type* LLVM_DtoSize_t();

void LLVM_DtoMain();

void LLVM_DtoCallClassDtors(TypeClass* tc, llvm::Value* instance);
void LLVM_DtoInitClass(TypeClass* tc, llvm::Value* dst);

llvm::Constant* LLVM_DtoConstInitializer(Type* type, Initializer* init);
void LLVM_DtoInitializer(Initializer* init);

llvm::Function* LLVM_DeclareMemSet32();
llvm::Function* LLVM_DeclareMemSet64();
llvm::Function* LLVM_DeclareMemCpy32();
llvm::Function* LLVM_DeclareMemCpy64();

llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* LLVM_DtoGEPi(llvm::Value* ptr, unsigned i0, const std::string& var, llvm::BasicBlock* bb=NULL);
llvm::Value* LLVM_DtoGEPi(llvm::Value* ptr, unsigned i0, unsigned i1, const std::string& var, llvm::BasicBlock* bb=NULL);

void LLVM_DtoGiveArgumentStorage(elem* e);

llvm::Value* LLVM_DtoRealloc(llvm::Value* ptr, const llvm::Type* ty);
llvm::Value* LLVM_DtoRealloc(llvm::Value* ptr, llvm::Value* len);

void LLVM_DtoAssert(llvm::Value* cond, llvm::Value* loc, llvm::Value* msg);

llvm::Value* LLVM_DtoArgument(const llvm::Type* paramtype, Argument* fnarg, Expression* argexp);

llvm::Value* LLVM_DtoNestedVariable(VarDeclaration* vd);

void LLVM_DtoAssign(Type* lhsType, llvm::Value* lhs, llvm::Value* rhs);

llvm::ConstantInt* LLVM_DtoConstSize_t(size_t);
llvm::ConstantInt* LLVM_DtoConstUint(unsigned i);
llvm::Constant* LLVM_DtoConstString(const char*);

void LLVM_DtoMemCpy(llvm::Value* dst, llvm::Value* src, llvm::Value* nbytes);

llvm::Value* LLVM_DtoIndexStruct(llvm::Value* ptr, StructDeclaration* sd, Type* t, unsigned os, std::vector<unsigned>& idxs);

#include "enums.h"
