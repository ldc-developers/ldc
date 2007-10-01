// D -> LLVM helpers

struct StructInitializer;

const llvm::Type* LLVM_DtoType(Type* t);

llvm::Type* LLVM_DtoStructType(Type* t);
llvm::Value* LLVM_DtoStructZeroInit(TypeStruct* t, llvm::Value* v);
llvm::Value* LLVM_DtoStructCopy(TypeStruct* t, llvm::Value* dst, llvm::Value* src);
llvm::Constant* LLVM_DtoStructInitializer(StructInitializer* si);

llvm::FunctionType* LLVM_DtoFunctionType(Type* t, const llvm::Type* thisparam = 0);
llvm::FunctionType* LLVM_DtoFunctionType(FuncDeclaration* fdecl);
llvm::Function* LLVM_DtoDeclareFunction(FuncDeclaration* fdecl);

llvm::StructType* LLVM_DtoDelegateType(Type* t);
llvm::Value* LLVM_DtoNullDelegate(llvm::Value* v);
llvm::Value* LLVM_DtoDelegateCopy(llvm::Value* dst, llvm::Value* src);

llvm::GlobalValue::LinkageTypes LLVM_DtoLinkage(PROT prot, uint stc);
unsigned LLVM_DtoCallingConv(LINK l);

llvm::Value* LLVM_DtoPointedType(llvm::Value* ptr, llvm::Value* val);
llvm::Value* LLVM_DtoBoolean(llvm::Value* val);

const llvm::Type* LLVM_DtoSize_t();

void LLVM_DtoMain();

void LLVM_DtoCallClassDtors(TypeClass* tc, llvm::Value* instance);
void LLVM_DtoInitClass(TypeClass* tc, llvm::Value* dst);

llvm::Constant* LLVM_DtoInitializer(Type* type, Initializer* init);

llvm::Function* LLVM_DeclareMemSet32();
llvm::Function* LLVM_DeclareMemSet64();
llvm::Function* LLVM_DeclareMemCpy32();
llvm::Function* LLVM_DeclareMemCpy64();

llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, llvm::Value* i0, llvm::Value* i1, const std::string& var, llvm::BasicBlock* bb);
llvm::Value* LLVM_DtoGEP(llvm::Value* ptr, const std::vector<unsigned>& src, const std::string& var, llvm::BasicBlock* bb);

#include "enums.h"
