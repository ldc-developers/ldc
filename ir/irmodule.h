#ifndef LLVMDC_IR_IRMODULE_H
#define LLVMDC_IR_IRMODULE_H

#include "ir/ir.h"

struct Module;

struct IrModule : IrBase
{
    IrModule(Module* module);
    virtual ~IrModule();

    Module* M;

    llvm::GlobalVariable* dwarfCompileUnit;
};

#endif
