//===-- dvalue.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dvalue.h"
#include "declaration.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DVarValue::DVarValue(Type* t, VarDeclaration* vd, LLValue* llvmValue)
: DValue(t), var(vd), val(llvmValue)
{
    assert(isaPointer(llvmValue));
    assert(!isSpecialRefVar(vd) ||
        isaPointer(isaPointer(llvmValue)->getElementType()));
}

DVarValue::DVarValue(Type* t, LLValue* llvmValue)
: DValue(t), var(0), val(llvmValue)
{
    assert(isaPointer(llvmValue));
}

LLValue* DVarValue::getLVal()
{
    assert(val);
    if (var && isSpecialRefVar(var))
        return DtoLoad(val);
    return val;
}

LLValue* DVarValue::getRVal()
{
    assert(val);
    Type* bt = type->toBasetype();

    LLValue* tmp = val;
    if (var && isSpecialRefVar(var))
        tmp = DtoLoad(tmp);

    if (DtoIsPassedByRef(bt))
        return tmp;
    return DtoLoad(tmp);
}

LLValue* DVarValue::getRefStorage()
{
    assert(val);
    assert(isSpecialRefVar(var));
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

LLValue* DSliceValue::getRVal()
{
    assert(len);
    assert(ptr);
    return DtoAggrPair(len, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(FuncDeclaration* fd, LLValue* v, LLValue* vt)
: DValue(fd->type), func(fd), val(v), vthis(vt)
{}

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
