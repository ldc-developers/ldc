// this head contains stuff used by all the IR

#ifndef LLVMDC_IR_IR_H
#define LLVMDC_IR_IR_H

#include "ir/irforw.h"
#include "root.h"

struct IrBase : Object
{
    virtual ~IrBase() {}
};

#endif
