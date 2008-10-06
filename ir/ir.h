// this head contains stuff used by all the IR

#ifndef LDC_IR_IR_H
#define LDC_IR_IR_H

#include "ir/irforw.h"
#include "root.h"

struct IrBase : Object
{
    virtual ~IrBase() {}
};

#endif
