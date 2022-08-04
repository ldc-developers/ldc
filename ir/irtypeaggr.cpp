//===-- irtypeaggr.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irtypeaggr.h"

#include "dmd/aggregate.h"
#include "dmd/errors.h"
#include "dmd/init.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"
#include "llvm/IR/DerivedTypes.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// FIXME A similar function is in ir/iraggr.cpp and RTTIBuilder::push().
static inline size_t add_zeros(std::vector<llvm::Type *> &defaultTypes,
                               size_t startOffset, size_t endOffset) {
  assert(startOffset <= endOffset);
  const size_t paddingSize = endOffset - startOffset;
  if (paddingSize) {
    llvm::ArrayType *pad = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(gIR->context()), paddingSize);
    defaultTypes.push_back(pad);
  }
  return paddingSize ? 1 : 0;
}

bool var_offset_sort_cb(const VarDeclaration *v1, const VarDeclaration *v2) {
  return v1->offset < v2->offset;
}

AggrTypeBuilder::AggrTypeBuilder(bool packed, unsigned offset)
    : m_offset(offset), m_packed(packed) {
  m_defaultTypes.reserve(32);
}

void AggrTypeBuilder::addType(llvm::Type *type, unsigned size) {
  m_defaultTypes.push_back(type);
  m_offset += size;
  m_fieldIndex++;
}

void AggrTypeBuilder::addAggregate(AggregateDeclaration *ad) {
  addAggregate(ad, nullptr, Aliases::AddToVarGEPIndices);
}

void AggrTypeBuilder::addAggregate(
    AggregateDeclaration *ad, const AggrTypeBuilder::VarInitMap *explicitInits,
    AggrTypeBuilder::Aliases aliases) {
  const size_t n = ad->fields.length;
  if (n == 0)
    return;

  // Unions may lead to overlapping fields, and we need to flatten them for LLVM
  // IR. We usually take the first field (in declaration order) of an
  // overlapping set, but a literal with an explicit initializer for a dominated
  // field might require us to select that field.
  LLSmallVector<VarDeclaration *, 16> actualFields;

  // Bit fields additionally complicate matters. E.g.:
  // struct S {
  //   unsigned char ub:6;
  //   _Bool b:1;
  //   unsigned ui:25;
  // };
  auto getFieldEnd = [](VarDeclaration *vd) {
    auto bf = vd->isBitFieldDeclaration();
    return vd->offset +
           (bf ? (bf->bitOffset + bf->fieldWidth + 7) / 8 : vd->type->size());
  };
  auto getFieldType = [](VarDeclaration *vd) -> LLType * {
    if (auto bf = vd->isBitFieldDeclaration()) {
      const auto sizeInBytes = (bf->bitOffset + bf->fieldWidth + 7) / 8;
      return LLIntegerType::get(gIR->context(), sizeInBytes * 8);
    }
    return DtoMemType(vd->type);
  };
  LLSmallVector<BitFieldDeclaration *, 16> allBitFieldDecls;
  for (VarDeclaration *field : ad->fields) {
    if (auto bf = field->isBitFieldDeclaration()) {
      //printf(".: %s: byte offset %d, bit offset %d, type size %d\n", bf->toChars(), bf->offset, bf->bitOffset, (int) bf->type->size());
      allBitFieldDecls.push_back(bf);
    }
  }
  std::map<VarDeclaration *, BitFieldDeclaration *> bitFieldsMap;
  if (!allBitFieldDecls.empty()) {
    // TODO: this flattens all bit fields, even if overlapping via unions
    // could probably be done properly in a single pass, grouping *consecutive* bit fields only
    std::sort(allBitFieldDecls.begin(), allBitFieldDecls.end(),
              [](BitFieldDeclaration *a, BitFieldDeclaration *b) {
                // 1) ascendingly by start byte offset
                if (a->offset != b->offset)
                  return a->offset < b->offset;
                // 2) descendingly by end bit offset
                return a->bitOffset + a->fieldWidth >
                       b->bitOffset + b->fieldWidth;
              });

    BitFieldDeclaration *currentDominantField = nullptr;
    for (BitFieldDeclaration *bf : allBitFieldDecls) {
      if (!currentDominantField) {
        currentDominantField = bf;
      } else if (bf->offset > currentDominantField->offset) {
        const auto currentEnd = getFieldEnd(currentDominantField);
        assert(currentEnd <= bf->offset); // might break for unions
        currentDominantField = bf;
      }
      bitFieldsMap[bf] = currentDominantField;
    }
  }

  // list of pairs: alias => actual field (same offset, same LL type)
  LLSmallVector<std::pair<VarDeclaration *, VarDeclaration *>, 16> aliasPairs;

  // Iterate over all fields in declaration order, in 1 or 2 passes.
  for (int pass = explicitInits ? 0 : 1; pass < 2; ++pass) {
    for (VarDeclaration *field : ad->fields) {
      // 1st pass: only for fields with explicit initializer
      if (pass == 0 && explicitInits->find(field) == explicitInits->end())
        continue;

      // 2nd pass: only for fields without explicit initializer
      if (pass == 1 && explicitInits && explicitInits->find(field) != explicitInits->end())
        continue;

      // skip dominated bit fields
      auto it = bitFieldsMap.find(field);
      if (it != bitFieldsMap.end() && it->second != field)
        continue;

      const size_t f_begin = field->offset;
      const size_t f_end = getFieldEnd(field);

      // skip empty fields
      if (f_begin == f_end)
        continue;

      // check for overlap with existing fields (on a byte level, not bits)
      bool overlaps = false;
      if (field->overlapped() || field->isBitFieldDeclaration()) {
        for (const auto existing : actualFields) {
          const size_t e_begin = existing->offset;
          const size_t e_end = getFieldEnd(existing);

          if (e_begin < f_end && e_end > f_begin) {
            overlaps = true;
            if (aliases == Aliases::AddToVarGEPIndices && e_begin == f_begin &&
                getFieldType(existing) == getFieldType(field)) {
              aliasPairs.push_back(std::make_pair(field, existing));
            }
            break;
          }
        }
      }

      if (!overlaps)
        actualFields.push_back(field);
    }
  }

  // Now we can build a list of LLVM types for the actual LL fields.
  // Make sure to zero out any padding and set the GEP indices for the directly
  // indexable variables.

  // first we sort the list by offset
  std::sort(actualFields.begin(), actualFields.end(), var_offset_sort_cb);

  for (const auto vd : actualFields) {
    assert(vd->offset >= m_offset);

    // Add an explicit field for any padding so we can zero it, as per TDPL
    // §7.1.1.
    if (m_offset < vd->offset) {
      m_fieldIndex += add_zeros(m_defaultTypes, m_offset, vd->offset);
      m_offset = vd->offset;
    }

    // add default type
    LLType *fieldType = getFieldType(vd);
    m_defaultTypes.push_back(fieldType);

    // advance offset to right past this field
    if (!fieldType->isSized()) {
      // forward reference in a cycle or similar, we need to trust the D type
      m_offset += vd->type->size();
    } else {
      m_offset += getTypeAllocSize(fieldType);
    }

    // set the field index
    m_varGEPIndices[vd] = m_fieldIndex;

    // let any aliases reuse this field/GEP index
    for (const auto &pair : aliasPairs) {
      if (pair.second == vd)
        m_varGEPIndices[pair.first] = m_fieldIndex;
    }

    // map the dominated other bit fields to this field/GEP index too
    if (vd->isBitFieldDeclaration()) {
      for (const auto &pair : bitFieldsMap) {
        if (pair.second == vd && pair.first != vd)
          m_varGEPIndices[pair.first] = m_fieldIndex;
      }
    }

    ++m_fieldIndex;
  }
}

void AggrTypeBuilder::alignCurrentOffset(unsigned alignment) {
  m_overallAlignment = std::max(alignment, m_overallAlignment);

  unsigned aligned = (m_offset + alignment - 1) & ~(alignment - 1);
  if (m_offset < aligned) {
    m_fieldIndex += add_zeros(m_defaultTypes, m_offset, aligned);
    m_offset = aligned;
  }
}

void AggrTypeBuilder::addTailPadding(unsigned aggregateSize) {
  if (m_offset < aggregateSize)
    add_zeros(m_defaultTypes, m_offset, aggregateSize);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration *ad)
    : IrType(ad->type,
             LLStructType::create(gIR->context(), ad->toPrettyChars())),
      aggr(ad) {}

bool IrTypeAggr::isPacked(AggregateDeclaration *ad) {
  // If the aggregate's size is unknown, any field with type alignment > 1 will
  // make it packed.
  unsigned aggregateSize = ~0u;
  unsigned aggregateAlignment = 1;
  if (ad->sizeok == Sizeok::done) {
    aggregateSize = ad->structsize;
    aggregateAlignment = ad->alignsize;

    if (auto sd = ad->isStructDeclaration()) {
      if (!sd->alignment.isDefault() && !sd->alignment.isPack())
        aggregateAlignment = sd->alignment.get();
    }
  }

  // Classes apparently aren't padded; their size may not match the alignment.
  assert((ad->isClassDeclaration() ||
          (aggregateSize & (aggregateAlignment - 1)) == 0) &&
         "Size not a multiple of alignment?");

  // For unions, only a subset of the fields are actually used for the IR type -
  // don't care (about a few potentially needlessly packed IR structs).
  for (const auto field : ad->fields) {
    // The aggregate size, aggregate alignment and the field offset need to be
    // multiples of the field type's alignment, otherwise the aggregate type is
    // unnaturally aligned, and LLVM would insert padding.
    const unsigned fieldTypeAlignment = DtoAlignment(field->type);
    const auto mask = fieldTypeAlignment - 1;
    if ((aggregateSize | aggregateAlignment | field->offset) & mask)
      return true;
  }

  return false;
}

void IrTypeAggr::getMemberLocation(VarDeclaration *var, unsigned &fieldIndex,
                                   unsigned &byteOffset) const {
  // Note: The interface is a bit more general than what we actually return.
  // Specifically, the frontend offset information we use for overlapping
  // fields is always based at the object start.
  auto it = varGEPIndices.find(var);
  if (it != varGEPIndices.end()) {
    fieldIndex = it->second;
    byteOffset = 0;
  } else {
    fieldIndex = 0;
    byteOffset = var->offset;
  }
}
