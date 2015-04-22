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

typedef std::map<VarDeclaration*, unsigned> VarGEPIndices;

class AggrTypeBuilder
{
public:
    AggrTypeBuilder(bool packed);
    void addType(llvm::Type *type, unsigned size);
    void addAggregate(AggregateDeclaration *ad);
    void alignCurrentOffset(unsigned alignment);
    void addTailPadding(unsigned aggregateSize);
    unsigned currentFieldIndex() const { return m_fieldIndex; }
    std::vector<llvm::Type*> defaultTypes() const { return m_defaultTypes; }
    VarGEPIndices varGEPIndices() const { return m_varGEPIndices; }
protected:
    std::vector<llvm::Type*> m_defaultTypes;
    VarGEPIndices m_varGEPIndices;
    unsigned m_offset;
    unsigned m_fieldIndex;
    bool m_packed;
};

/// Base class of IrTypes for aggregate types.
class IrTypeAggr : public IrType
{
public:
    ///
    IrTypeAggr* isAggr()            { return this; }

    /// Returns the index of the field in the LLVM struct type that corresponds
    /// to the given member variable, plus the offset to the actual field start
    /// due to overlapping (union) fields, if any.
    void getMemberLocation(VarDeclaration* var, unsigned& fieldIndex,
        unsigned& byteOffset) const;

    /// Composite type debug description. This is not only to cache, but also
    /// used for resolving forward references.
#if LDC_LLVM_VER >= 307
    llvm::MDType* diCompositeType = nullptr;
#else
    llvm::DIType diCompositeType;
#endif

    /// true, if the LLVM struct type for the aggregate is declared as packed
    bool packed;

protected:
    ///
    IrTypeAggr(AggregateDeclaration* ad);

    /// Returns true, if the LLVM struct type for the aggregate must be declared
    /// as packed.
    static bool isPacked(AggregateDeclaration* ad);

    /// AggregateDeclaration this type represents.
    AggregateDeclaration* aggr;

    /// Stores the mapping from member variables to field indices in the actual
    /// LLVM type. If a member variable is not present, this means that it does
    /// not resolve to a "clean" GEP but extra offsetting due to overlapping
    /// members is needed (i.e., a union).
    ///
    /// We need to keep track of this separately, because there is no way to get
    /// the field index of a variable in the frontend, it only stores the byte
    /// offset.
    std::map<VarDeclaration*, unsigned> varGEPIndices;
};

#endif
