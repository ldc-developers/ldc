#include "gen/llvm.h"
#include "ir/irmodule.h"

IrModule::IrModule(Module* module)
{
    M = module;
    dwarfCompileUnit = NULL;
}

IrModule::~IrModule()
{
}
