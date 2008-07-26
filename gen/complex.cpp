#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"

#include "gen/complex.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoComplexType(Type* type)
{
    Type* t = DtoDType(type);

    const LLType* base = DtoComplexBaseType(t);

    std::vector<const LLType*> types;
    types.push_back(base);
    types.push_back(base);

    return llvm::StructType::get(types);
}

const LLType* DtoComplexBaseType(Type* t)
{
    TY ty = DtoDType(t)->ty;
    const LLType* base;
    if (ty == Tcomplex32) {
        return LLType::FloatTy;
    }
    else if (ty == Tcomplex64) {
        return LLType::DoubleTy;
    }
    else if (ty == Tcomplex80) {
        return (global.params.useFP80) ? LLType::X86_FP80Ty : LLType::DoubleTy;
    }
    else {
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstComplex(Type* ty, LLConstant* re, LLConstant* im)
{
    assert(0);
    const LLType* base = DtoComplexBaseType(ty);

    std::vector<LLConstant*> inits;
    inits.push_back(re);
    inits.push_back(im);

    const llvm::VectorType* vt = llvm::VectorType::get(base, 2);
    return llvm::ConstantVector::get(vt, inits);
}

LLConstant* DtoConstComplex(Type* _ty, long double re, long double im)
{
    TY ty = DtoDType(_ty)->ty;

    llvm::ConstantFP* fre;
    llvm::ConstantFP* fim;

    Type* base = 0;

    if (ty == Tcomplex32) {
        base = Type::tfloat32;
    }
    else if (ty == Tcomplex64) {
        base = Type::tfloat64;
    }
    else if (ty == Tcomplex80) {
        base = (global.params.useFP80) ? Type::tfloat80 : Type::tfloat64;
    }

    std::vector<LLConstant*> inits;
    inits.push_back(DtoConstFP(base, re));
    inits.push_back(DtoConstFP(base, im));

    return llvm::ConstantStruct::get(DtoComplexType(_ty), inits);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoRealPart(DValue* val)
{
    assert(0);
    return gIR->ir->CreateExtractElement(val->getRVal(), DtoConstUint(0), "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoImagPart(DValue* val)
{
    assert(0);
    return gIR->ir->CreateExtractElement(val->getRVal(), DtoConstUint(1), "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplex(Loc& loc, Type* to, DValue* val)
{
    Type* t = DtoDType(val->getType());

    if (val->isComplex() || t->iscomplex()) {
        return DtoCastComplex(loc, val, to);
    }

    const LLType* base = DtoComplexBaseType(to);

    Type* baserety;
    Type* baseimty;
    TY ty = to->ty;
    if (ty == Tcomplex32) {
        baserety = Type::tfloat32;
        baseimty = Type::timaginary32;
    } else if (ty == Tcomplex64) {
        baserety = Type::tfloat64;
        baseimty = Type::timaginary64;
    } else if (ty == Tcomplex80) {
        baserety = global.params.useFP80 ? Type::tfloat80 : Type::tfloat64;
        baseimty = global.params.useFP80 ? Type::timaginary80 : Type::timaginary64;
    }

    if (t->isimaginary()) {
        return new DComplexValue(to, LLConstant::getNullValue(DtoType(baserety)), DtoCastFloat(loc, val, baseimty)->getRVal());
    }
    else if (t->isfloating()) {
        return new DComplexValue(to, DtoCastFloat(loc, val, baserety)->getRVal(), LLConstant::getNullValue(DtoType(baseimty)));
    }
    else if (t->isintegral()) {
        return new DComplexValue(to, DtoCastInt(loc, val, baserety)->getRVal(), LLConstant::getNullValue(DtoType(baseimty)));
    }
    assert(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoComplexAssign(LLValue* l, LLValue* r)
{
    DtoStore(DtoLoad(DtoGEPi(r, 0,0, "tmp")), DtoGEPi(l,0,0,"tmp"));
    DtoStore(DtoLoad(DtoGEPi(r, 0,1, "tmp")), DtoGEPi(l,0,1,"tmp"));
}

void DtoComplexSet(LLValue* c, LLValue* re, LLValue* im)
{
    DtoStore(re, DtoGEPi(c,0,0,"tmp"));
    DtoStore(im, DtoGEPi(c,0,1,"tmp"));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoGetComplexParts(DValue* c, LLValue*& re, LLValue*& im)
{
    // get LLValues
    if (DComplexValue* cx = c->isComplex()) {
        re = cx->re;
        im = cx->im;
    }
    else {
        re = DtoLoad(DtoGEPi(c->getRVal(),0,0,"tmp"));
        im = DtoLoad(DtoGEPi(c->getRVal(),0,1,"tmp"));
    }
}

DValue* resolveLR(DValue* val, bool getlval)
{
    if (DLRValue* lr = val->isLRValue()) {
        if (getlval)
            return lr->lvalue;
        else
            return lr->rvalue;
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexAdd(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(loc, type, resolveLR(lhs, true));
    rhs = DtoComplex(loc, type, resolveLR(rhs, false));

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

DValue* DtoComplexSub(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(loc, type, resolveLR(lhs, true));
    rhs = DtoComplex(loc, type, resolveLR(rhs, false));

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

DValue* DtoComplexMul(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(loc, type, resolveLR(lhs, true));
    rhs = DtoComplex(loc, type, resolveLR(rhs, false));

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

DValue* DtoComplexDiv(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    lhs = DtoComplex(loc, type, resolveLR(lhs, true));
    rhs = DtoComplex(loc, type, resolveLR(rhs, false));

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

DValue* DtoComplexNeg(Loc& loc, Type* type, DValue* val)
{
    val = DtoComplex(loc, type, resolveLR(val, false));

    llvm::Value *a, *b, *re, *im;

    // values
    DtoGetComplexParts(val, a, b);

    // sub up
    re = gIR->ir->CreateNeg(a, "tmp");
    im = gIR->ir->CreateNeg(b, "tmp");

    return new DComplexValue(type, re, im);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoComplexEquals(Loc& loc, TOK op, DValue* lhs, DValue* rhs)
{
    Type* type = lhs->getType();

    lhs = DtoComplex(loc, type, resolveLR(lhs, false));
    rhs = DtoComplex(loc, type, resolveLR(rhs, false));

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

    // (l.re==r.re && l.im==r.im) or (l.re!=r.re || l.im!=r.im)
    LLValue* b1 = new llvm::FCmpInst(cmpop, a, c, "tmp", gIR->scopebb());
    LLValue* b2 = new llvm::FCmpInst(cmpop, b, d, "tmp", gIR->scopebb());

    if (op == TOKequal)
        return gIR->ir->CreateAnd(b1,b2,"tmp");
    else
        return gIR->ir->CreateOr(b1,b2,"tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastComplex(Loc& loc, DValue* val, Type* _to)
{
    Type* to = DtoDType(_to);
    Type* vty = val->getType();
    if (to->iscomplex()) {
        if (vty->size() == to->size())
            return val;

        llvm::Value *re, *im;
        DtoGetComplexParts(val, re, im);
        const LLType* toty = DtoComplexBaseType(to);

        if (to->size() < vty->size()) {
            re = gIR->ir->CreateFPTrunc(re, toty, "tmp");
            im = gIR->ir->CreateFPTrunc(im, toty, "tmp");
        }
        else if (to->size() > vty->size()) {
            re = gIR->ir->CreateFPExt(re, toty, "tmp");
            im = gIR->ir->CreateFPExt(im, toty, "tmp");
        }
        else {
            return val;
        }

        if (val->isComplex())
            return new DComplexValue(_to, re, im);

        // unfortunately at this point, the cast value can show up as the lvalue for += and similar expressions.
        // so we need to give it storage, or fix the system that handles this stuff (DLRValue)
        LLValue* mem = new llvm::AllocaInst(DtoType(_to), "castcomplextmp", gIR->topallocapoint());
        DtoComplexSet(mem, re, im);
        return new DLRValue(val, new DImValue(_to, mem));
    }
    else if (to->isimaginary()) {
        if (val->isComplex())
            return new DImValue(to, val->isComplex()->im);
        LLValue* v = val->getRVal();
        DImValue* im = new DImValue(to, DtoLoad(DtoGEPi(v,0,1,"tmp")));
        return DtoCastFloat(loc, im, to);
    }
    else if (to->isfloating()) {
        if (val->isComplex())
            return new DImValue(to, val->isComplex()->re);
        LLValue* v = val->getRVal();
        DImValue* re = new DImValue(to, DtoLoad(DtoGEPi(v,0,0,"tmp")));
        return DtoCastFloat(loc, re, to);
    }
    else
    assert(0);
}

