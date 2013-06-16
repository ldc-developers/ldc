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
#if LDC_LLVM_VER >= 302
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

struct AggregateDeclaration;
struct VarDeclaration;

/// Base class of IrTypes for aggregate types.
class IrTypeAggr : public IrType
{
public:
    ///
    IrTypeAggr* isAggr()            { return this; }

    ///
    typedef std::vector<VarDeclaration*>::iterator iterator;

    ///
    iterator def_begin()        { return default_fields.begin(); }

    ///
    iterator def_end()          { return default_fields.end(); }


    /// Composite type debug description. This is not only to cache, but also
    /// used for resolving forward references.
    llvm::DIType diCompositeType;

protected:
    ///
    IrTypeAggr(AggregateDeclaration* ad);

    /// AggregateDeclaration this type represents.
    AggregateDeclaration* aggr;

    /// Sorted list of all default fields.
    /// A default field is a field that contributes to the default initializer
    /// and the default type, and thus it has it's own unique GEP index into
    /// the aggregate.
    /// For classes, field of any super classes are not included.
    std::vector<VarDeclaration*> default_fields;
};

#endif
