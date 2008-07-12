#include "gen/llvm.h"

#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/dvalue.h"

#include "declaration.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DVarValue::DVarValue(VarDeclaration* vd, LLValue* llvmValue, bool lvalue)
{
    var = vd;
    val = llvmValue;
    rval = 0;
    lval = lvalue;
    type = var->type;
}

DVarValue::DVarValue(Type* t, LLValue* lv, LLValue* rv)
{
    var = 0;
    val = lv;
    rval = rv;
    lval = true;
    type = t;
}

DVarValue::DVarValue(Type* t, LLValue* llvmValue, bool lvalue)
{
    var = 0;
    val = llvmValue;
    rval = 0;
    lval = lvalue;
    type = t;
}

LLValue* DVarValue::getLVal()
{
    assert(val && lval);
    return val;
}

LLValue* DVarValue::getRVal()
{
    assert(rval || val);
    if (DtoIsPassedByRef(type)) {
        if (rval) return rval;
        return val;
    }
    else {
        if (rval) return rval;
        //Logger::cout() << "val: " << *val << '\n';
        if (!isThis() && !isField() && DtoCanLoad(val)) {
            return DtoLoad(val);
        }
        return val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(FuncDeclaration* fd, LLValue* v, LLValue* vt)
{
    func = fd;
    type = func->type;
    val = v;
    vthis = vt;
    cc = (unsigned)-1;
}

LLValue* DFuncValue::getRVal()
{
    assert(val);
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

LLValue* DConstValue::getRVal()
{
    assert(c);
    return c;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
