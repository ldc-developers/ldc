//===-- ir/irtypeaggr.h - IrType subclasses for aggregates ------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRTYPEAGGR_H
#define LDC_IR_IRTYPEAGGR_H

#include "ir/irtype.h"
#include "llvm/ADT/ArrayRef.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/DebugInfo.h"
#elif LDC_LLVM_VER >= 302
#include "llvm/DebugInfo.h"
#else
#include "llvm/Analysis/DebugInfo.h"
#endif
#include <map>
#include <vector>

namespace llvm {
    class Constant;
    class StructType;
}

class AggregateDeclaration;
class VarDeclaration;

/// Base class of IrTypes for aggregate types.
class IrTypeAggr : public IrType
{
public:
    ///
    IrTypeAggr* isAggr()            { return this; }

    /// Composite type debug description. This is not only to cache, but also
    /// used for resolving forward references.
    llvm::DIType diCompositeType;

protected:
    ///
    IrTypeAggr(AggregateDeclaration* ad);

    /// AggregateDeclaration this type represents.
    AggregateDeclaration* aggr;
};

#endif
