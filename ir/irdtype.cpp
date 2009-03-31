#include "gen/llvm.h"
#include "ir/ir.h"
#include "ir/irdtype.h"

std::set<IrDType*> IrDType::list;

void IrDType::resetAll()
{
    std::set<IrDType*>::iterator it;
    for(it = list.begin(); it != list.end(); ++it)
        (*it)->reset();
}

IrDType::IrDType()
{
    assert(list.insert(this).second);
    reset();
}

IrDType::IrDType(const IrDType& s)
{
    assert(list.insert(this).second);
    type = s.type;
}

IrDType::~IrDType()
{
    list.erase(this);
}

void IrDType::reset()
{
    type = NULL;
}
