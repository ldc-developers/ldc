#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "gen/irstate.h"
#include "ir/irmodule.h"

IrModule::IrModule(Module* module, const char* srcfilename)
{
    M = module;

    LLConstant* slice = DtoConstString(srcfilename);
    fileName = new llvm::GlobalVariable(
        *gIR->module, slice->getType(), true, LLGlobalValue::InternalLinkage, slice, ".modulefilename");
}

IrModule::~IrModule()
{
}
