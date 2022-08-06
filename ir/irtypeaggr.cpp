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
  //   unsigned char a:7; // byte offset 0, bit offset 0, bit width 7
  //   _Bool b:1;         // byte offset 0, bit offset 7, bit width 1
  //   _Bool c:1;         // byte offset 1, bit offset 0, bit width 1
  //   unsigned ui:23;    // byte offset 1, bit offset 1, bit width 23
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

  // list of pairs: alias => actual field (same offset, same LL type)
  LLSmallVector<std::pair<VarDeclaration *, VarDeclaration *>, 16> aliasPairs;

  // Iterate over all fields in declaration order, in 1 or 2 passes.
  for (int pass = explicitInits ? 0 : 1; pass < 2; ++pass) {
    for (size_t i = 0; i < ad->fields.length; ++i) {
      VarDeclaration *field = ad->fields[i];
      bool haveExplicitInit =
          explicitInits && explicitInits->find(field) != explicitInits->end();

      const auto bf = field->isBitFieldDeclaration();
      if (bf) {
        // group all consecutive bit fields at the same byte offset (and with
        // non-overlapping bits)
        const auto startIndex = i;
        size_t j; // endIndex
        unsigned bitEnd = bf->bitOffset + bf->fieldWidth;
        for (j = i + 1; j < ad->fields.length; ++j) {
          if (auto bf2 = ad->fields[j]->isBitFieldDeclaration()) {
            if (bf2->offset == bf->offset && bf2->bitOffset >= bitEnd) {
              bitEnd = bf2->bitOffset + bf2->fieldWidth;
              if (!haveExplicitInit) {
                haveExplicitInit = explicitInits && explicitInits->find(bf2) !=
                                                        explicitInits->end();
              }
              continue;
            }
          }
          break;
        }

        // use the last bit field (with the highest bits) for the whole group
        i = j - 1; // skip the other bit fields in the fields loop
        field = ad->fields[i];

        // create unconditional aliases for the other bit fields in the group
        for (size_t k = startIndex; k < i; ++k) {
          aliasPairs.push_back(std::make_pair(ad->fields[k], field));
        }
      }

      // 1st pass: only for fields with explicit initializer
      if (pass == 0 && !haveExplicitInit)
        continue;

      // 2nd pass: only for fields without explicit initializer
      if (pass == 1 && haveExplicitInit)
        continue;

      const size_t f_begin = field->offset;
      const size_t f_end = getFieldEnd(field);

      // skip empty fields
      if (f_begin == f_end)
        continue;

      // check for overlap with existing fields (on a byte level, not bits)
      bool overlaps = false;
      if (bf || field->overlapped()) {
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
    if (vd->offset < m_offset) {
      vd->error(
          "overlaps previous field. This is an ICE, please file an LDC issue.");
      fatal();
    }

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
      const auto llSize = getTypeAllocSize(fieldType);
      assert(llSize <= vd->type->size() || vd->isBitFieldDeclaration());
      m_offset += llSize;
    }

    // set the field index
    m_varGEPIndices[vd] = m_fieldIndex;

    // let any aliases reuse this field/GEP index
    for (const auto &pair : aliasPairs) {
      if (pair.second == vd)
        m_varGEPIndices[pair.first] = m_fieldIndex;
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
  assert(m_offset <= aggregateSize &&
         "IR aggregate type is larger than the corresponding D type");
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
