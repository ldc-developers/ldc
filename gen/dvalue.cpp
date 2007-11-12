#include "gen/llvm.h"

#include "declaration.h"

#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/dvalue.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DVarValue::DVarValue(VarDeclaration* vd, llvm::Value* llvmValue, bool lvalue)
{
    var = vd;
    val = llvmValue;
    rval = 0;
    lval = lvalue;
    type = var->type;
}

DVarValue::DVarValue(Type* t, llvm::Value* lv, llvm::Value* rv)
{
    var = 0;
    val = lv;
    rval = rv;
    lval = true;
    type = t;
}

DVarValue::DVarValue(Type* t, llvm::Value* llvmValue, bool lvalue)
{
    var = 0;
    val = llvmValue;
    rval = 0;
    lval = lvalue;
    type = t;
}

llvm::Value* DVarValue::getLVal()
{
    assert(val && lval);
    return val;
}

llvm::Value* DVarValue::getRVal()
{
    assert(rval || val);
    if (DtoIsPassedByRef(type)) {
        if (rval) return rval;
        return val;
    }
    else {
        if (rval) return rval;
        Logger::cout() << "val: " << *val << '\n';
        if (isaArgument(val)) {
            if (var && (var->isRef() || var->isOut()))
                return DtoLoad(val);
        }
        else if (!isField() && DtoCanLoad(val)) {
            return DtoLoad(val);
        }
        return val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(FuncDeclaration* fd, llvm::Value* v, llvm::Value* vt)
{
    func = fd;
    type = func->type;
    val = v;
    vthis = vt;
    cc = (unsigned)-1;
}

llvm::Value* DFuncValue::getLVal()
{
    assert(0);
    return 0;
}

llvm::Value* DFuncValue::getRVal()
{
    assert(val);
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DConstValue::getRVal()
{
    assert(c);
    return c;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
