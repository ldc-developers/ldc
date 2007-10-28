#ifndef LLVMC_GEN_ARRAYS_H
#define LLVMC_GEN_ARRAYS_H

const llvm::StructType* LLVM_DtoArrayType(Type* t);
const llvm::ArrayType* LLVM_DtoStaticArrayType(Type* t);

llvm::Constant* LLVM_DtoConstArrayInitializer(ArrayInitializer* si);
llvm::Constant* LLVM_DtoConstSlice(llvm::Constant* dim, llvm::Constant* ptr);
llvm::Constant* LLVM_DtoConstStaticArray(const llvm::Type* t, llvm::Constant* c);

void LLVM_DtoArrayCopy(elem* dst, elem* src);
void LLVM_DtoArrayInit(llvm::Value* l, llvm::Value* r);
void LLVM_DtoArrayInit(llvm::Value* ptr, llvm::Value* dim, llvm::Value* val);
void LLVM_DtoArrayAssign(llvm::Value* l, llvm::Value* r);
void LLVM_DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr);
void LLVM_DtoNullArray(llvm::Value* v);

llvm::Value* LLVM_DtoNewDynArray(llvm::Value* dst, llvm::Value* dim, Type* dty, bool doinit=true);
void LLVM_DtoResizeDynArray(llvm::Value* arr, llvm::Value* sz);

void LLVM_DtoCatAssignElement(llvm::Value* arr, Expression* exp);
void LLVM_DtoCatArrays(llvm::Value* arr, Expression* e1, Expression* e2);

void LLVM_DtoStaticArrayCopy(llvm::Value* dst, llvm::Value* src);

llvm::Value* LLVM_DtoStaticArrayCompare(TOK op, llvm::Value* l, llvm::Value* r);

llvm::Value* LLVM_DtoDynArrayCompare(TOK op, llvm::Value* l, llvm::Value* r);
llvm::Value* LLVM_DtoDynArrayIs(TOK op, llvm::Value* l, llvm::Value* r);

llvm::Value* LLVM_DtoArrayCastLength(llvm::Value* len, const llvm::Type* elemty, const llvm::Type* newelemty);

#endif // LLVMC_GEN_ARRAYS_H
