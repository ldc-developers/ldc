#ifndef LLVMDC_GEN_COMPLEX_H
#define LLVMDC_GEN_COMPLEX_H

const llvm::StructType* DtoComplexType(Type* t);
const llvm::Type* DtoComplexBaseType(Type* t);

llvm::Constant* DtoConstComplex(Type* t, llvm::Constant* re, llvm::Constant* im);
llvm::Constant* DtoConstComplex(Type* t, long double re, long double im);
llvm::Constant* DtoUndefComplex(Type* _ty);

llvm::Constant* DtoComplexShuffleMask(unsigned a, unsigned b);

llvm::Value* DtoRealPart(DValue* val);
llvm::Value* DtoImagPart(DValue* val);
DValue* DtoComplex(Type* to, DValue* val);

void DtoComplexAssign(llvm::Value* l, llvm::Value* r);
void DtoComplexSet(llvm::Value* c, llvm::Value* re, llvm::Value* im);

DValue* DtoComplexAdd(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexSub(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexMul(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexDiv(Type* type, DValue* lhs, DValue* rhs);

llvm::Value* DtoComplexEquals(TOK op, DValue* lhs, DValue* rhs);

#endif // LLVMDC_GEN_COMPLEX_H
