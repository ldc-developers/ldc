#include "gen/llvm.h"
#include "ir/ir.h"
#include "ir/irtype.h"

std::set<IrType*> IrType::list;

void IrType::resetAll()
{
    std::set<IrType*>::iterator it;
    for(it = list.begin(); it != list.end(); ++it)
        (*it)->reset();
}

IrType::IrType()
{
    assert(list.insert(this).second);
    reset();
}

IrType::IrType(const IrType& s)
{
    assert(list.insert(this).second);
    type = s.type;
    vtblType = s.type;
}

IrType::~IrType()
{
    list.erase(this);
}

void IrType::reset()
{
    type = NULL;
    vtblType = NULL;
}
