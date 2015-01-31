//===-- irmodule.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "module.h"
#include "gen/llvm.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
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
