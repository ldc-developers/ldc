#ifndef LLVMDC_GEN_COMPLEX_H
#define LLVMDC_GEN_COMPLEX_H

const llvm::StructType* DtoComplexType(Type* t);
const llvm::Type* DtoComplexBaseType(Type* t);

LLConstant* DtoConstComplex(Type* t, LLConstant* re, LLConstant* im);
LLConstant* DtoConstComplex(Type* t, long double re, long double im);
LLConstant* DtoUndefComplex(Type* _ty);

LLConstant* DtoComplexShuffleMask(unsigned a, unsigned b);

LLValue* DtoRealPart(DValue* val);
LLValue* DtoImagPart(DValue* val);
DValue* DtoComplex(Type* to, DValue* val);

void DtoComplexAssign(LLValue* l, LLValue* r);
void DtoComplexSet(LLValue* c, LLValue* re, LLValue* im);

void DtoGetComplexParts(DValue* c, LLValue*& re, LLValue*& im);

DValue* DtoComplexAdd(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexSub(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexMul(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexDiv(Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexNeg(Type* type, DValue* val);

LLValue* DtoComplexEquals(TOK op, DValue* lhs, DValue* rhs);

#endif // LLVMDC_GEN_COMPLEX_H
