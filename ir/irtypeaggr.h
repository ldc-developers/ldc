//===-- ir/irtypeaggr.h - IrType subclasses for aggregates ------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ir/irtype.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DebugInfo.h"
#include <map>
#include <vector>

namespace llvm {
class Constant;
class StructType;
}

class AggregateDeclaration;
class BitFieldDeclaration;
class VarDeclaration;

using VarGEPIndices = std::map<VarDeclaration *, unsigned>;
using ExtraBitFieldMembers =
    std::vector<std::pair<BitFieldDeclaration *, BitFieldDeclaration *>>;

class AggrTypeBuilder {
public:
  using VarInitMap = std::map<VarDeclaration *, llvm::Constant *>;
  enum class Aliases { Skip, AddToVarGEPIndices };

  explicit AggrTypeBuilder(unsigned offset = 0);
  void addType(llvm::Type *type, unsigned size);
  void addAggregate(AggregateDeclaration *ad);
  void addAggregate(AggregateDeclaration *ad, const VarInitMap *explicitInits,
                    Aliases aliases);
  void alignCurrentOffset(unsigned alignment);
  void addTailPadding(unsigned aggregateSize);
  unsigned currentFieldIndex() const { return m_fieldIndex; }
  const std::vector<llvm::Type *> &defaultTypes() const {
    return m_defaultTypes;
  }
  const VarGEPIndices &varGEPIndices() const { return m_varGEPIndices; }
  const ExtraBitFieldMembers &extraBitFieldMembers() const {
    return m_extraBitFieldMembers;
  }
  bool isPacked() const { return m_packed; }
  unsigned currentOffset() const { return m_offset; }

protected:
  std::vector<llvm::Type *> m_defaultTypes;
  VarGEPIndices m_varGEPIndices;
  // list of pairs: extra bit field member (greater byte offset) => first member
  // of bit field group
  ExtraBitFieldMembers m_extraBitFieldMembers;
  unsigned m_offset = 0;
  unsigned m_fieldIndex = 0;
  unsigned m_maxFieldIRAlignment = 1;
  bool m_packed = false; // in IR terms
};

/// Base class of IrTypes for aggregate types.
class IrTypeAggr : public IrType {
public:
  ///
  IrTypeAggr *isAggr() override { return this; }

  /// Returns the index of the field in the LLVM struct type that corresponds
  /// to the given member variable, plus the offset to the actual field start
  /// due to overlapping (union) fields, if any.
  unsigned getMemberLocation(VarDeclaration *var, bool& isFieldIdx);

protected:
  ///
  explicit IrTypeAggr(AggregateDeclaration *ad);

  /// AggregateDeclaration this type represents.
  AggregateDeclaration *aggr = nullptr;

  /// Stores the mapping from member variables to field indices in the actual
  /// LLVM type. If a member variable is not present, this means that it does
  /// not resolve to a "clean" GEP but extra offsetting due to overlapping
  /// members is needed (i.e., a union).
  ///
  /// We need to keep track of this separately, because there is no way to get
  /// the field index of a variable in the frontend, it only stores the byte
  /// offset.
  VarGEPIndices varGEPIndices;

  virtual const VarGEPIndices &getVarGEPIndices() { return varGEPIndices; }
};

// A helper for aggregating consecutive bit fields to a group.
struct BitFieldGroup {
  unsigned byteOffset = 0;  // from aggregate start
  unsigned sizeInBytes = 0; // to cover the highest bit
  llvm::SmallVector<BitFieldDeclaration *, 8> bitFields;

  static BitFieldGroup
  startingFrom(size_t startFieldIndex, size_t numTotalFields,
               std::function<VarDeclaration *(size_t i)> getFieldFn);

  unsigned getBitOffset(BitFieldDeclaration *member) const;
};
