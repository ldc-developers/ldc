//===-- irtypeaggr.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irtypeaggr.h"

#if LDC_LLVM_VER >= 303
#include "llvm/IR/DerivedTypes.h"
#else
#include "llvm/DerivedTypes.h"
#endif

#include "aggregate.h"

#include "gen/irstate.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration * ad)
:   IrType(ad->type, LLStructType::create(gIR->context(), ad->toPrettyChars())),
    diCompositeType(NULL), aggr(ad)
{
}

void IrTypeAggr::getMemberLocation(VarDeclaration* var, unsigned& fieldIndex,
    unsigned& byteOffset) const
{
    // Note: The interface is a bit more general than what we actually return.
    // Specifically, the frontend offset information we use for overlapping
    // fields is always based at the object start.
    std::map<VarDeclaration*, unsigned>::const_iterator it =
        varGEPIndices.find(var);
    if (it != varGEPIndices.end())
    {
        fieldIndex = it->second;
        byteOffset = 0;
    }
    else
    {
        fieldIndex = 0;
        byteOffset = var->offset;
    }
}
