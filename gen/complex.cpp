#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"

#include "gen/complex.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"
#include "gen/logger.h"

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoComplexType(Type* type)
{
    Type* t = type->toBasetype();
    const LLType* base = DtoComplexBaseType(t);
    return llvm::StructType::get(base, base, NULL);
}

const LLType* DtoComplexBaseType(Type* t)
{
    TY ty = t->toBasetype()->ty;
    const LLType* base;
    if (ty == Tcomplex32) {
        return LLType::FloatTy;
    }
    else if (ty == Tcomplex64) {
        return LLType::DoubleTy;
    }
    else if (ty == Tcomplex80) {
        if (global.params.cpu == ARCHx86)
            return LLType::X86_FP80Ty;
        else
            return LLType::DoubleTy;
    }
    else {
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstComplex(Type* _ty, long double re, long double im)
{
    TY ty = _ty->toBasetype()->ty;

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
        base = Type::tfloat80;
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
    Type* t = val->getType()->toBasetype();

    if (t->iscomplex()) {
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
        baserety = Type::tfloat80;
        baseimty = Type::timaginary80;
    }
    else {
        assert(0);
    }

    const LLType* complexTy = DtoType(to);
    LLValue* res;

    if (t->isimaginary()) {
        res = DtoAggrPair(complexTy, LLConstant::getNullValue(DtoType(baserety)), DtoCastFloat(loc, val, baseimty)->getRVal());
    }
    else if (t->isfloating()) {
        res = DtoAggrPair(complexTy, DtoCastFloat(loc, val, baserety)->getRVal(), LLConstant::getNullValue(DtoType(baseimty)));
    }
    else if (t->isintegral()) {
        res = DtoAggrPair(complexTy, DtoCastInt(loc, val, baserety)->getRVal(), LLConstant::getNullValue(DtoType(baseimty)));
    }
    else {
        assert(0);
    }
    return new DImValue(to, res);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoComplexSet(LLValue* c, LLValue* re, LLValue* im)
{
    DtoStore(re, DtoGEPi(c,0,0,"tmp"));
    DtoStore(im, DtoGEPi(c,0,1,"tmp"));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoGetComplexParts(DValue* c, LLValue*& re, LLValue*& im)
{
    LLValue* v = c->getRVal();
    Logger::cout() << "extracting real and imaginary parts from: " << *v << '\n';
    re = gIR->ir->CreateExtractValue(v, 0, ".re_part");
    im = gIR->ir->CreateExtractValue(v, 1, ".im_part");
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

bool hasRe(Type* t)
{
    return 
        (t->ty != Timaginary32 && 
         t->ty != Timaginary64 &&
         t->ty != Timaginary80);
}

bool hasIm(Type* t)
{
    return 
        (t->ty == Timaginary32 || 
         t->ty == Timaginary64 ||
         t->ty == Timaginary80 ||
         t->ty == Tcomplex32 || 
         t->ty == Tcomplex64 ||
         t->ty == Tcomplex80);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexAdd(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    DValue* clhs = DtoComplex(loc, type, resolveLR(lhs, true));
    DValue* crhs = DtoComplex(loc, type, resolveLR(rhs, false));

    llvm::Value *lhs_re, *lhs_im, *rhs_re, *rhs_im, *res_re, *res_im;

    // lhs values
    DtoGetComplexParts(clhs, lhs_re, lhs_im);
    // rhs values
    DtoGetComplexParts(crhs, rhs_re, rhs_im);

    // add up
    Type* lhstype = lhs->getType();
    Type* rhstype = rhs->getType();
    if(hasRe(lhstype) && hasRe(rhstype))
        res_re = gIR->ir->CreateAdd(lhs_re, rhs_re, "tmp");
    else if(hasRe(lhstype))
        res_re = lhs_re;
    else // either hasRe(rhstype) or no re at all (then use any)
        res_re = rhs_re;
    
    if(hasIm(lhstype) && hasIm(rhstype))
        res_im = gIR->ir->CreateAdd(lhs_im, rhs_im, "tmp");
    else if(hasIm(lhstype))
        res_im = lhs_im;
    else // either hasIm(rhstype) or no im at all (then use any)
        res_im = rhs_im;

    LLValue* res = DtoAggrPair(DtoType(type), res_re, res_im);
    return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexSub(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    DValue* clhs = DtoComplex(loc, type, resolveLR(lhs, true));
    DValue* crhs = DtoComplex(loc, type, resolveLR(rhs, false));

    llvm::Value *lhs_re, *lhs_im, *rhs_re, *rhs_im, *res_re, *res_im;

    // lhs values
    DtoGetComplexParts(clhs, lhs_re, lhs_im);
    // rhs values
    DtoGetComplexParts(crhs, rhs_re, rhs_im);

    // sub up
    Type* lhstype = lhs->getType();
    Type* rhstype = rhs->getType();
    if(hasRe(rhstype))
        res_re = gIR->ir->CreateSub(lhs_re, rhs_re, "tmp");
    else
        res_re = lhs_re;
    
    if(hasIm(rhstype))
        res_im = gIR->ir->CreateSub(lhs_im, rhs_im, "tmp");
    else
        res_im = lhs_im;

    LLValue* res = DtoAggrPair(DtoType(type), res_re, res_im);
    return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexMul(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    DValue* clhs = DtoComplex(loc, type, resolveLR(lhs, true));
    DValue* crhs = DtoComplex(loc, type, resolveLR(rhs, false));

    llvm::Value *lhs_re, *lhs_im, *rhs_re, *rhs_im, *res_re, *res_im;

    // lhs values
    DtoGetComplexParts(clhs, lhs_re, lhs_im);
    // rhs values
    DtoGetComplexParts(crhs, rhs_re, rhs_im);

    // mul up
    llvm::Value *rere = NULL;
    llvm::Value *reim = NULL;
    llvm::Value *imre = NULL;
    llvm::Value *imim = NULL;
    Type* lhstype = lhs->getType();
    Type* rhstype = rhs->getType();

    if(hasRe(lhstype) && hasRe(rhstype))
        rere = gIR->ir->CreateMul(lhs_re, rhs_re, "rere_mul");
    if(hasRe(lhstype) && hasIm(rhstype))
        reim = gIR->ir->CreateMul(lhs_re, rhs_im, "reim_mul");
    if(hasIm(lhstype) && hasRe(rhstype))
        imre = gIR->ir->CreateMul(lhs_im, rhs_re, "imre_mul");
    if(hasIm(lhstype) && hasIm(rhstype))
        imim = gIR->ir->CreateMul(lhs_im, rhs_im, "imim_mul");

    if(rere && imim)
        res_re = gIR->ir->CreateSub(rere, imim, "rere_imim_sub");
    else if(rere)
        res_re = rere;
    else if(imim)
        res_re = gIR->ir->CreateNeg(imim, "imim_neg");
    else
        res_re = hasRe(lhstype) ? rhs_re : lhs_re; // null!

    if(reim && imre)
        res_im = gIR->ir->CreateAdd(reim, imre, "reim_imre_add");
    else if(reim)
        res_im = reim;
    else if(imre)
        res_im = imre;
    else
        res_im = hasRe(lhstype) ? rhs_im : lhs_re; // null!

    LLValue* res = DtoAggrPair(DtoType(type), res_re, res_im);
    return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoComplexDiv(Loc& loc, Type* type, DValue* lhs, DValue* rhs)
{
    DValue* clhs = DtoComplex(loc, type, resolveLR(lhs, true));
    DValue* crhs = DtoComplex(loc, type, resolveLR(rhs, false));

    llvm::Value *lhs_re, *lhs_im, *rhs_re, *rhs_im, *res_re, *res_im;

    // lhs values
    DtoGetComplexParts(clhs, lhs_re, lhs_im);
    // rhs values
    DtoGetComplexParts(crhs, rhs_re, rhs_im);

    Type* lhstype = lhs->getType();
    Type* rhstype = rhs->getType();

    // if divisor is only real, division is simple
    if(hasRe(rhstype) && !hasIm(rhstype)) {
        if(hasRe(lhstype))
            res_re = gIR->ir->CreateFDiv(lhs_re, rhs_re, "re_divby_re");
        else
            res_re = lhs_re;
        if(hasIm(lhstype))
            res_im = gIR->ir->CreateFDiv(lhs_im, rhs_re, "im_divby_re");
        else
            res_im = lhs_im;
    }
    // if divisor is only imaginary, division is simple too
    else if(!hasRe(rhstype) && hasIm(rhstype)) {
        if(hasRe(lhstype))
            res_im = gIR->ir->CreateNeg(gIR->ir->CreateFDiv(lhs_re, rhs_im, "re_divby_im"), "neg");
        else
            res_im = lhs_re;
        if(hasIm(lhstype))
            res_re = gIR->ir->CreateFDiv(lhs_im, rhs_im, "im_divby_im");
        else
            res_re = lhs_im;
    }
    // full division
    else {
        llvm::Value *tmp1, *tmp2, *denom;

        if(hasRe(lhstype) && hasIm(lhstype)) {
            tmp1 = gIR->ir->CreateMul(lhs_re, rhs_re, "rere");
            tmp2 = gIR->ir->CreateMul(lhs_im, rhs_im, "imim");
            res_re = gIR->ir->CreateAdd(tmp1, tmp2, "rere_plus_imim");

            tmp1 = gIR->ir->CreateMul(lhs_re, rhs_im, "reim");
            tmp2 = gIR->ir->CreateMul(lhs_im, rhs_re, "imre");
            res_im = gIR->ir->CreateSub(tmp2, tmp1, "imre_sub_reim");
        }
        else if(hasRe(lhstype)) {
            res_re = gIR->ir->CreateMul(lhs_re, rhs_re, "rere");

            res_im = gIR->ir->CreateMul(lhs_re, rhs_im, "reim");
            res_im = gIR->ir->CreateNeg(res_im);
        }
        else if(hasIm(lhstype)) {
            res_re = gIR->ir->CreateMul(lhs_im, rhs_im, "imim");
            res_im = gIR->ir->CreateMul(lhs_im, rhs_re, "imre");
        }
        else
            assert(0 && "lhs has neither real nor imaginary part");

        tmp1 = gIR->ir->CreateMul(rhs_re, rhs_re, "rhs_resq");
        tmp2 = gIR->ir->CreateMul(rhs_im, rhs_im, "rhs_imsq");
        denom = gIR->ir->CreateAdd(tmp1, tmp2, "denom");

        res_re = gIR->ir->CreateFDiv(res_re, denom, "res_re");
        res_im = gIR->ir->CreateFDiv(res_im, denom, "res_im");
    }

    LLValue* res = DtoAggrPair(DtoType(type), res_re, res_im);
    return new DImValue(type, res);
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

    LLValue* res = DtoAggrPair(DtoType(type), re, im);
    return new DImValue(type, res);
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
    Type* to = _to->toBasetype();
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
        else {
            re = gIR->ir->CreateFPExt(re, toty, "tmp");
            im = gIR->ir->CreateFPExt(im, toty, "tmp");
        }

        LLValue* pair = DtoAggrPair(DtoType(_to), re, im);
        DValue* rval = new DImValue(_to, pair);

        // if the value we're casting is not a lvalue, the cast value can't be either
        if (!val->isLVal()) {
            return rval;
        }

        // unfortunately at this point, the cast value can show up as the lvalue for += and similar expressions.
        // so we need to maintain the storage
        return new DLRValue(val, rval);
    }
    else if (to->isimaginary()) {
        // FIXME: this loads both values, even when we only need one
        LLValue* v = val->getRVal();
        LLValue* impart = gIR->ir->CreateExtractValue(v, 1, ".im_part");
        DImValue* im = new DImValue(to, impart);
        return DtoCastFloat(loc, im, to);
    }
    else if (to->isfloating()) {
        // FIXME: this loads both values, even when we only need one
        LLValue* v = val->getRVal();
        LLValue* repart = gIR->ir->CreateExtractValue(v, 0, ".re_part");
        DImValue* re = new DImValue(to, repart);
        return DtoCastFloat(loc, re, to);
    }
    else
    assert(0);
}

