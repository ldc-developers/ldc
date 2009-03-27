// this head contains stuff used by all the IR

#ifndef LDC_IR_IR_H
#define LDC_IR_IR_H

#include "ir/irforw.h"
#include "root.h"

struct IRState;

struct IrBase : Object
{
    virtual ~IrBase() {}
};

struct Ir
{
    Ir() : irs(NULL) {}

    void setState(IRState* p)   { irs = p; }
    IRState* getState()         { return irs; }

private:
    IRState* irs;
};

#endif
