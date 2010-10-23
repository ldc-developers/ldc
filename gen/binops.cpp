#include "gen/llvm.h"

#include "declaration.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/dvalue.h"
#include "gen/logger.h"
#include "gen/complex.h"

//////////////////////////////////////////////////////////////////////////////

DValue* DtoBinAdd(DValue* lhs, DValue* rhs)
{
    Type* t = lhs->getType();
    LLValue *l, *r;
    l = lhs->getRVal();
    r = rhs->getRVal();
    assert(l->isintegral == r->isintegral);
    
    LLValue* res;
    if (t->isfloating())
        res = gIR->ir->CreateFAdd(l, r, "tmp");
    else
        res = gIR->ir->CreateAdd(l, r, "tmp");
    
    return new DImValue( t, res );
}

//////////////////////////////////////////////////////////////////////////////

DValue* DtoBinSub(DValue* lhs, DValue* rhs)
{
    Type* t = lhs->getType();
    LLValue *l, *r;
    l = lhs->getRVal();
    r = rhs->getRVal();
    assert(l->isintegral == r->isintegral);
    
    LLValue* res;
    if (t->isfloating())
        res = gIR->ir->CreateFSub(l, r, "tmp");
    else
        res = gIR->ir->CreateSub(l, r, "tmp");
    
    return new DImValue( t, res );
}

//////////////////////////////////////////////////////////////////////////////

DValue* DtoBinMul(Type* targettype, DValue* lhs, DValue* rhs)
{
    Type* t = lhs->getType();
    LLValue *l, *r;
    l = lhs->getRVal();
    r = rhs->getRVal();
    assert(l->isintegral == r->isintegral);
    
    LLValue* res;
    if (t->isfloating())
        res = gIR->ir->CreateFMul(l, r, "tmp");
    else
        res = gIR->ir->CreateMul(l, r, "tmp");
    return new DImValue( targettype, res );
}

//////////////////////////////////////////////////////////////////////////////

DValue* DtoBinDiv(Type* targettype, DValue* lhs, DValue* rhs)
{
    Type* t = lhs->getType();
    LLValue *l, *r;
    l = lhs->getRVal();
    r = rhs->getRVal();
    
    LLValue* res;
    if (t->isfloating())
        res = gIR->ir->CreateFDiv(l, r, "tmp");
    else if (!t->isunsigned())
        res = gIR->ir->CreateSDiv(l, r, "tmp");
    else
        res = gIR->ir->CreateUDiv(l, r, "tmp");
    return new DImValue( targettype, res );
}

//////////////////////////////////////////////////////////////////////////////

DValue* DtoBinRem(Type* targettype, DValue* lhs, DValue* rhs)
{
    Type* t = lhs->getType();
    LLValue *l, *r;
    l = lhs->getRVal();
    r = rhs->getRVal();
    LLValue* res;
    if (t->isfloating())
        res = gIR->ir->CreateFRem(l, r, "tmp");
    else if (!t->isunsigned())
        res = gIR->ir->CreateSRem(l, r, "tmp");
    else
        res = gIR->ir->CreateURem(l, r, "tmp");
    return new DImValue( targettype, res );
}

//////////////////////////////////////////////////////////////////////////////

LLValue* DtoBinNumericEquals(Loc loc, DValue* lhs, DValue* rhs, TOK op)
{
    assert(op == TOKequal || op == TOKnotequal ||
           op == TOKidentity || op == TOKnotidentity);
    Type* t = lhs->getType()->toBasetype();
    assert(t->isfloating());
    Logger::println("numeric equality");

    LLValue* lv = lhs->getRVal();
    LLValue* rv = rhs->getRVal();
    LLValue* res = 0;

    if (t->iscomplex())
    {
        Logger::println("complex");
        res = DtoComplexEquals(loc, op, lhs, rhs);
    }
    else if (t->isfloating())
    {
        Logger::println("floating");
        res = (op == TOKidentity || op == TOKequal)
        ?   gIR->ir->CreateFCmpOEQ(lv,rv,"tmp")
        :   gIR->ir->CreateFCmpUNE(lv,rv,"tmp");
    }
    
    assert(res);
    return res;
}
