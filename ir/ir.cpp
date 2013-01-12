//===-- ir.cpp ------------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 303
#include "llvm/IR/DataLayout.h"
#elif LDC_LLVM_VER == 302
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/functions.h"

#include "ir/ir.h"
#include "ir/irfunction.h"


unsigned GetTypeAlignment(Ir* ir, Type* t)
{
    return gDataLayout->getABITypeAlignment(DtoType(t));
}

unsigned GetPointerSize(Ir* ir)
{
    return gDataLayout->getPointerSize(ADDRESS_SPACE);
}

unsigned GetTypeStoreSize(Ir* ir, Type* t)
{
    return gDataLayout->getTypeStoreSize(DtoType(t));
}

unsigned GetTypeAllocSize(Ir* ir, Type* t)
{
    return gDataLayout->getTypeAllocSize(DtoType(t));
}

Ir::Ir()
: irs(NULL)
{
}

void Ir::addFunctionBody(IrFunction * f)
{
    functionbodies.push_back(f);
}

void Ir::emitFunctionBodies()
{
    while (!functionbodies.empty())
    {
        IrFunction* irf = functionbodies.front();
        functionbodies.pop_front();
        DtoDefineFunction(irf->decl);
    }
}
