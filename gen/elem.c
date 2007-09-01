#include <iostream>

#include "llvm/Instructions.h"

#include "elem.h"

#include "irstate.h"
#include "logger.h"

//////////////////////////////////////////////////////////////////////////////////////////

elem::elem()
{
    mem = 0;
    val = 0;
    arg = 0;

    type = NONE;
    inplace = false;
    field = false;

    vardecl = 0;
    funcdecl = 0;
}

llvm::Value* elem::getValue()
{
    assert(val || mem);
    switch(type)
    {
    case NONE:
        assert(0 && "type == NONE");
        break;

    case VAR:
    case REF: {
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
                return new llvm::LoadInst(mem, "tmp", gIR->scopebb());
            }
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
