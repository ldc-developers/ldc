#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"

#include "gen/complex.h"
#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoComplexType(Type* type)
{
    Type* t = DtoDType(type);

    const llvm::Type* base = DtoComplexBaseType(t);

    std::vector<const llvm::Type*> types;
    types.push_back(base);
    types.push_back(base);

    return llvm::StructType::get(types);
}

const llvm::Type* DtoComplexBaseType(Type* t)
{
    TY ty = DtoDType(t)->ty;
    const llvm::Type* base;
    if (ty == Tcomplex32) {
        return llvm::Type::FloatTy;
    }
    else if (ty == Tcomplex64 || ty == Tcomplex80) {
        return llvm::Type::DoubleTy;
    }
    else {
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* DtoConstComplex(Type* ty, llvm::Constant* re, llvm::Constant* im)
{
    assert(0);
    const llvm::Type* base = DtoComplexBaseType(ty);

    std::vector<llvm::Constant*> inits;
    inits.push_back(re);
    inits.push_back(im);

    const llvm::VectorType* vt = llvm::VectorType::get(base, 2);
    return llvm::ConstantVector::get(vt, inits);
}

llvm::Constant* DtoConstComplex(Type* _ty, long double re, long double im)
{
    TY ty = DtoDType(_ty)->ty;

    llvm::ConstantFP* fre;
    llvm::ConstantFP* fim;

    const llvm::Type* base;

    if (ty == Tcomplex32) {
        fre = DtoConstFP(Type::tfloat32, re);
        fim = DtoConstFP(Type::tfloat32, im);
        base = llvm::Type::FloatTy;
    }
    else if (ty == Tcomplex64 || ty == Tcomplex80) {
        fre = DtoConstFP(Type::tfloat64, re);
        fim = DtoConstFP(Type::tfloat64, im);
        base = llvm::Type::DoubleTy;
    }
    else
    assert(0);

    std::vector<llvm::Constant*> inits;
    inits.push_back(fre);
    inits.push_back(fim);
    return llvm::ConstantStruct::get(DtoComplexType(_ty), inits);
}

llvm::Constant* DtoUndefComplex(Type* _ty)
{
    assert(0);
    TY ty = DtoDType(_ty)->ty;
    const llvm::Type* base;
    if (ty == Tcomplex32) {
        base = llvm::Type::FloatTy;
    }
    else if (ty == Tcomplex64 || ty == Tcomplex80) {
        base = llvm::Type::DoubleTy;
    }
    else
    assert(0);

    std::vector<llvm::Constant*> inits;
    inits.push_back(llvm::UndefValue::get(base));
    inits.push_back(llvm::UndefValue::get(base));

    const llvm::VectorType* vt = llvm::VectorType::get(base, 2);
    return llvm::ConstantVector::get(vt, inits);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoRealPart(DValue* val)
{
    assert(0);
    return gIR->ir->CreateExtractElement(val->getRVal(), DtoConstUint(0), "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoImagPart(DValue* val)
{
    assert(0);
    return gIR->ir->CreateExtractElement(val->getRVal(), DtoConstUint(1), "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplex(Type* to, DValue* val)
{
    Type* t = DtoDType(val->getType());
    TY ty = t->ty;

    if (val->isComplex() || t->iscomplex()) {
        return DtoCastComplex(val, to);
    }

    const llvm::Type* base = DtoComplexBaseType(to);

    llvm::Constant* undef = llvm::UndefValue::get(base);
    llvm::Constant* zero;
    if (ty == Tfloat32 || ty == Timaginary32 || ty == Tcomplex32)
        zero = llvm::ConstantFP::get(llvm::Type::FloatTy, llvm::APFloat(0.0f));
    else if (ty == Tfloat64 || ty == Timaginary64 || ty == Tcomplex64 || ty == Tfloat80 || ty == Timaginary80 || ty == Tcomplex80)
        zero = llvm::ConstantFP::get(llvm::Type::DoubleTy, llvm::APFloat(0.0));

    if (t->isimaginary()) {
        return new DComplexValue(to, zero, val->getRVal());
    }
    else if (t->isfloating()) {
        return new DComplexValue(to, val->getRVal(), zero);
    }
    else
    assert(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoComplexAssign(llvm::Value* l, llvm::Value* r)
{
    DtoStore(DtoLoad(DtoGEPi(r, 0,0, "tmp")), DtoGEPi(l,0,0,"tmp"));
    DtoStore(DtoLoad(DtoGEPi(r, 0,1, "tmp")), DtoGEPi(l,0,1,"tmp"));
}

void DtoComplexSet(llvm::Value* c, llvm::Value* re, llvm::Value* im)
{
    DtoStore(re, DtoGEPi(c,0,0,"tmp"));
    DtoStore(im, DtoGEPi(c,0,1,"tmp"));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoGetComplexParts(DValue* c, llvm::Value*& re, llvm::Value*& im)
{
    // lhs values
    if (DComplexValue* cx = c->isComplex()) {
        re = cx->re;
        im = cx->im;
    }
    else {
        re = DtoLoad(DtoGEPi(c->getRVal(),0,0,"tmp"));
        im = DtoLoad(DtoGEPi(c->getRVal(),0,1,"tmp"));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexAdd(Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(type, lhs);
    rhs = DtoComplex(type, rhs);

    llvm::Value *a, *b, *c, *d, *re, *im;

    // lhs values
    DtoGetComplexParts(lhs, a, b);
    // rhs values
    DtoGetComplexParts(rhs, c, d);

    // add up
    re = gIR->ir->CreateAdd(a, c, "tmp");
    im = gIR->ir->CreateAdd(b, d, "tmp");

    return new DComplexValue(type, re, im);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexSub(Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(type, lhs);
    rhs = DtoComplex(type, rhs);

    llvm::Value *a, *b, *c, *d, *re, *im;

    // lhs values
    DtoGetComplexParts(lhs, a, b);
    // rhs values
    DtoGetComplexParts(rhs, c, d);

    // add up
    re = gIR->ir->CreateSub(a, c, "tmp");
    im = gIR->ir->CreateSub(b, d, "tmp");

    return new DComplexValue(type, re, im);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexMul(Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(type, lhs);
    rhs = DtoComplex(type, rhs);

    llvm::Value *a, *b, *c, *d;

    // lhs values
    DtoGetComplexParts(lhs, a, b);
    // rhs values
    DtoGetComplexParts(rhs, c, d);

    llvm::Value *tmp1, *tmp2, *re, *im;

    tmp1 = gIR->ir->CreateMul(a, c, "tmp");
    tmp2 = gIR->ir->CreateMul(b, d, "tmp");
    re = gIR->ir->CreateSub(tmp1, tmp2, "tmp");

    tmp1 = gIR->ir->CreateMul(b, c, "tmp");
    tmp2 = gIR->ir->CreateMul(a, d, "tmp");
    im = gIR->ir->CreateAdd(tmp1, tmp2, "tmp");

    return new DComplexValue(type, re, im);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexDiv(Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(type, lhs);
    rhs = DtoComplex(type, rhs);

    llvm::Value *a, *b, *c, *d;

    // lhs values
    DtoGetComplexParts(lhs, a, b);
    // rhs values
    DtoGetComplexParts(rhs, c, d);

    llvm::Value *tmp1, *tmp2, *denom, *re, *im;

    tmp1 = gIR->ir->CreateMul(c, c, "tmp");
    tmp2 = gIR->ir->CreateMul(d, d, "tmp");
    denom = gIR->ir->CreateAdd(tmp1, tmp2, "tmp");

    tmp1 = gIR->ir->CreateMul(a, c, "tmp");
    tmp2 = gIR->ir->CreateMul(b, d, "tmp");
    re = gIR->ir->CreateAdd(tmp1, tmp2, "tmp");
    re = gIR->ir->CreateFDiv(re, denom, "tmp");

    tmp1 = gIR->ir->CreateMul(b, c, "tmp");
    tmp2 = gIR->ir->CreateMul(a, d, "tmp");
    im = gIR->ir->CreateSub(tmp1, tmp2, "tmp");
    im = gIR->ir->CreateFDiv(im, denom, "tmp");

    return new DComplexValue(type, re, im);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoComplexEquals(TOK op, DValue* lhs, DValue* rhs)
{
    Type* type = lhs->getType();

    lhs = DtoComplex(type, lhs);
    rhs = DtoComplex(type, rhs);

    llvm::Value *a, *b, *c, *d;

    // lhs values
    DtoGetComplexParts(lhs, a, b);
    // rhs values
    DtoGetComplexParts(rhs, c, d);

    // select predicate
    llvm::FCmpInst::Predicate cmpop;
    if (op == TOKequal)
        cmpop = llvm::FCmpInst::FCMP_OEQ;
    else
        cmpop = llvm::FCmpInst::FCMP_UNE;

    // (l.re==r.re && l.im==r.im)
    llvm::Value* b1 = new llvm::FCmpInst(cmpop, a, c, "tmp", gIR->scopebb());
    llvm::Value* b2 = new llvm::FCmpInst(cmpop, b, d, "tmp", gIR->scopebb());
    return gIR->ir->CreateAnd(b1,b2,"tmp");
}
