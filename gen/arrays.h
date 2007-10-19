#ifndef LLVMC_GEN_ARRAYS_H
#define LLVMC_GEN_ARRAYS_H

const llvm::StructType* LLVM_DtoArrayType(Type* t);
const llvm::ArrayType* LLVM_DtoStaticArrayType(Type* t);

llvm::Constant* LLVM_DtoConstArrayInitializer(ArrayInitializer* si);
llvm::Constant* LLVM_DtoConstantSlice(llvm::Constant* dim, llvm::Constant* ptr);

void LLVM_DtoArrayCopy(elem* dst, elem* src);
void LLVM_DtoArrayInit(llvm::Value* l, llvm::Value* r);
void LLVM_DtoArrayInit(llvm::Value* ptr, llvm::Value* dim, llvm::Value* val);
void LLVM_DtoArrayAssign(llvm::Value* l, llvm::Value* r);
void LLVM_DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr);
void LLVM_DtoNullArray(llvm::Value* v);

void LLVM_DtoNewDynArray(llvm::Value* dst, llvm::Value* dim, Type* dty, bool doinit=true);
void LLVM_DtoResizeDynArray(llvm::Value* arr, llvm::Value* sz);

void LLVM_DtoCatArrayElement(llvm::Value* arr, Expression* exp);

void LLVM_DtoStaticArrayCopy(llvm::Value* dst, llvm::Value* src);
llvm::Value* LLVM_DtoStaticArrayCompare(TOK op, llvm::Value* l, llvm::Value* r);

#endif // LLVMC_GEN_ARRAYS_H
