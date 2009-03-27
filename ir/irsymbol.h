#ifndef __LDC_IR_IRSYMBOL_H__
#define __LDC_IR_IRSYMBOL_H__

#include "ir/ir.h"

/// Base class for all symbols.
struct IrSymbol
{
    ///
    IrSymbol(Ir* ir) : ir(ir) {}

    /// Migrate symbols to current module if necessary.
    virtual void migrate() = 0;

protected:
    ///
    Ir* ir;
};

#endif
