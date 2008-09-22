#ifndef LLVMDC_GEN_COMPLEX_H
#define LLVMDC_GEN_COMPLEX_H

const llvm::StructType* DtoComplexType(Type* t);
const LLType* DtoComplexBaseType(Type* t);

LLConstant* DtoConstComplex(Type* t, long double re, long double im);

LLConstant* DtoComplexShuffleMask(unsigned a, unsigned b);

LLValue* DtoRealPart(DValue* val);
LLValue* DtoImagPart(DValue* val);
DValue* DtoComplex(Loc& loc, Type* to, DValue* val);

void DtoComplexSet(LLValue* c, LLValue* re, LLValue* im);

void DtoGetComplexParts(Loc& loc, Type* to, DValue* c, LLValue*& re, LLValue*& im);

DValue* DtoComplexAdd(Loc& loc, Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexSub(Loc& loc, Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexMul(Loc& loc, Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexDiv(Loc& loc, Type* type, DValue* lhs, DValue* rhs);
DValue* DtoComplexNeg(Loc& loc, Type* type, DValue* val);

LLValue* DtoComplexEquals(Loc& loc, TOK op, DValue* lhs, DValue* rhs);

DValue* DtoCastComplex(Loc& loc, DValue* val, Type* to);

#endif // LLVMDC_GEN_COMPLEX_H
