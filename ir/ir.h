// this head contains stuff used by all the IR

#ifndef LDC_IR_IR_H
#define LDC_IR_IR_H

#include <deque>

#include "ir/irforw.h"
#include "root.h"

struct IRState;
struct IrFunction;

struct IrBase : Object
{
    virtual ~IrBase() {}
};

class Ir
{
public:
    Ir();

    void setState(IRState* p)   { irs = p; }
    IRState* getState()         { return irs; }

    void addFunctionBody(IrFunction* f);
    void emitFunctionBodies();

private:
    IRState* irs;

    std::deque<IrFunction*> functionbodies;
};

#endif
