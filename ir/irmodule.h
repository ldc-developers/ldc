#ifndef LDC_IR_IRMODULE_H
#define LDC_IR_IRMODULE_H

#include "ir/ir.h"

struct Module;

struct IrModule : IrBase
{
    IrModule(Module* module, const char* srcfilename);
    virtual ~IrModule();

    Module* M;

    LLGlobalVariable* dwarfCompileUnit;
    LLGlobalVariable* fileName;
};

#endif
