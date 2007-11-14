#ifndef LLVMC_GEN_ARRAYS_H
#define LLVMC_GEN_ARRAYS_H

struct DSliceValue;

const llvm::StructType* DtoArrayType(Type* t);
const llvm::ArrayType* DtoStaticArrayType(Type* t);

llvm::Constant* DtoConstArrayInitializer(ArrayInitializer* si);
llvm::Constant* DtoConstSlice(llvm::Constant* dim, llvm::Constant* ptr);
llvm::Constant* DtoConstStaticArray(const llvm::Type* t, llvm::Constant* c);

void DtoArrayCopy(DSliceValue* dst, DSliceValue* src);

void DtoArrayInit(llvm::Value* l, llvm::Value* r);
void DtoArrayInit(llvm::Value* ptr, llvm::Value* dim, llvm::Value* val);
void DtoArrayAssign(llvm::Value* l, llvm::Value* r);
void DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr);
void DtoNullArray(llvm::Value* v);

llvm::Value* DtoNewDynArray(llvm::Value* dst, llvm::Value* dim, Type* dty, bool doinit=true);
llvm::Value* DtoResizeDynArray(llvm::Value* arr, llvm::Value* sz);

void DtoCatAssignElement(llvm::Value* arr, Expression* exp);
void DtoCatAssignArray(llvm::Value* arr, Expression* exp);
void DtoCatArrays(llvm::Value* arr, Expression* e1, Expression* e2);

void DtoStaticArrayCopy(llvm::Value* dst, llvm::Value* src);

llvm::Value* DtoArrayEquals(TOK op, DValue* l, DValue* r);

llvm::Value* DtoDynArrayIs(TOK op, llvm::Value* l, llvm::Value* r);

llvm::Value* DtoArrayCastLength(llvm::Value* len, const llvm::Type* elemty, const llvm::Type* newelemty);

llvm::Value* DtoArrayLen(DValue* v);
llvm::Value* DtoArrayPtr(DValue* v);

#endif // LLVMC_GEN_ARRAYS_H
