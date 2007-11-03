/*
#include <iostream>

#include "gen/llvm.h"

#include "gen/elem.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/dvalue.h"

//////////////////////////////////////////////////////////////////////////////////////////

elem::elem(Expression* e)
{
    exp = e;

    mem = 0;
    val = 0;
    arg = 0;

    type = NONE;
    inplace = false;
    field = false;
    callconv = (unsigned)-1;
    isthis = false;
    istypeinfo = false;
    temp = false;

    vardecl = 0;
    funcdecl = 0;

    dvalue = 0;
}

elem::~elem()
{
    delete dvalue;
}

llvm::Value* elem::getValue()
{
    if (dvalue && !dvalue->isSlice()) {
        Logger::println("HAS DVALUE");
        return dvalue->getRVal();
    }

    assert(val || mem);
    switch(type)
    {
    case NONE:
        assert(0 && "type == NONE");
        break;

    case VAR:
    case REF:
    case ARRAYLEN:
        if (val) {
            return val;
        }
        else {
            if (!llvm::isa<llvm::PointerType>(mem->getType()))
            {
                Logger::cout() << "unexpected type: " << *mem->getType() << '\n';
                assert(0);
            }
            const llvm::PointerType* pt = llvm::cast<llvm::PointerType>(mem->getType());
            if (!pt->getElementType()->isFirstClassType()) {
                return mem;
            }
            else {
                return gIR->ir->CreateLoad(mem, "tmp");
            }
        }

    case VAL:
    case NUL:
    case FUNC:
    case CONST:
    case SLICE:
        return val ? val : mem;
    }
    assert(0 && "type == invalid value");
    return 0;
}
*/
