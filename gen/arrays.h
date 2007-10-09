#ifndef LLVMC_GEN_ARRAYS_H
#define LLVMC_GEN_ARRAYS_H

const llvm::StructType* LLVM_DtoArrayType(Type* t);
const llvm::ArrayType* LLVM_DtoStaticArrayType(Type* t);

llvm::Constant* LLVM_DtoArrayInitializer(ArrayInitializer* si);
llvm::Constant* LLVM_DtoConstantSlice(llvm::Constant* dim, llvm::Constant* ptr);

void LLVM_DtoArrayCopy(elem* dst, elem* src);
void LLVM_DtoArrayInit(llvm::Value* l, llvm::Value* r);
void LLVM_DtoArrayAssign(llvm::Value* l, llvm::Value* r);
void LLVM_DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr);
llvm::Value* LLVM_DtoNullArray(llvm::Value* v);

void LLVM_DtoNewDynArray(llvm::Value* dst, llvm::Value* dim, const llvm::Type* ty);
void LLVM_DtoResizeDynArray(llvm::Value* arr, llvm::Value* sz);

#endif // LLVMC_GEN_ARRAYS_H
