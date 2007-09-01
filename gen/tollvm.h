// D -> LLVM helpers

struct StructInitializer;

const llvm::Type* LLVM_DtoType(Type* t);

llvm::Type* LLVM_DtoStructType(Type* t);
llvm::Value* LLVM_DtoStructZeroInit(TypeStruct* t, llvm::Value* v);
llvm::Value* LLVM_DtoStructCopy(TypeStruct* t, llvm::Value* dst, llvm::Value* src);
llvm::Constant* LLVM_DtoStructInitializer(StructInitializer* si);

llvm::FunctionType* LLVM_DtoFunctionType(Type* t, const llvm::Type* thisparam = 0);
llvm::FunctionType* LLVM_DtoFunctionType(FuncDeclaration* fdecl);

llvm::StructType* LLVM_DtoDelegateType(Type* t);
llvm::Value* LLVM_DtoNullDelegate(llvm::Value* v);
llvm::Value* LLVM_DtoDelegateCopy(llvm::Value* dst, llvm::Value* src);

llvm::StructType* LLVM_DtoArrayType(Type* t);
llvm::ArrayType* LLVM_DtoStaticArrayType(Type* t);
llvm::Value* LLVM_DtoNullArray(llvm::Value* v);
llvm::Value* LLVM_DtoArrayAssign(llvm::Value* l, llvm::Value* r);
void LLVM_DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr);
llvm::Constant* LLVM_DtoArrayInitializer(ArrayInitializer* si);
void LLVM_DtoArrayCopy(elem* dst, elem* src);

void LLVM_DtoArrayInit(llvm::Value* l, llvm::Value* r);

llvm::GlobalValue::LinkageTypes LLVM_DtoLinkage(PROT prot, uint stc);
unsigned LLVM_DtoCallingConv(LINK l);

llvm::Value* LLVM_DtoPointedType(llvm::Value* ptr, llvm::Value* val);
llvm::Value* LLVM_DtoBoolean(llvm::Value* val);

const llvm::Type* LLVM_DtoSize_t();

void LLVM_DtoMain();

void LLVM_DtoCallClassDtors(TypeClass* tc, llvm::Value* instance);
void LLVM_DtoInitClass(TypeClass* tc, llvm::Value* dst);

llvm::Constant* LLVM_DtoInitializer(Type* type, Initializer* init);

#include "enums.h"
