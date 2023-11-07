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

AggrTypeBuilder::AggrTypeBuilder(unsigned offset) : m_offset(offset) {
  m_defaultTypes.reserve(32);
}

void AggrTypeBuilder::addType(llvm::Type *type, unsigned size) {
  const unsigned fieldAlignment = getABITypeAlign(type);
  assert(fieldAlignment);
  assert((m_offset & (fieldAlignment - 1)) == 0 && "Field is misaligned");
  m_defaultTypes.push_back(type);
  m_offset += size;
  m_fieldIndex++;
  m_maxFieldIRAlignment = std::max(m_maxFieldIRAlignment, fieldAlignment);
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
  struct Data {
    VarDeclaration *field;
    LLType *llType;
    uint64_t size;
  };
  LLSmallVector<Data, 16> actualFields;

  // list of pairs: alias => actual field (same offset, same LL type (not
  // checked for bit fields))
  LLSmallVector<std::pair<VarDeclaration *, VarDeclaration *>, 16> aliasPairs;

  // Bit fields additionally complicate matters. E.g.:
  // struct S {
  //   unsigned char a:7; // byte offset 0, bit offset 0, bit width 7
  //   _Bool b:1;         // byte offset 0, bit offset 7, bit width 1
  //   _Bool c:1;         // byte offset 1, bit offset 0, bit width 1
  //   unsigned d:22;     // byte offset 1, bit offset 1, bit width 22
  //   _Bool e:1;         // byte offset 3, bit offset 7, bit width 1
  // };
  // => group 1: byte offset 0, size 1 (`a`, with alias `b`)
  //    group 2: byte offset 1, size 3 (`c`, with alias `d` and extra member `e`
  //             (with greater byte offset))

  // list of pairs: extra bit field member (greater byte offset) => first member
  // of bit field group
  LLSmallVector<std::pair<BitFieldDeclaration *, BitFieldDeclaration *>, 8>
      extraBitFieldMembers;

  // Iterate over all fields in declaration order, in 1 or 2 passes.
  for (int pass = explicitInits ? 0 : 1; pass < 2; ++pass) {
    for (size_t i = 0; i < ad->fields.length; ++i) {
      const auto field = ad->fields[i];

      bool haveExplicitInit =
          explicitInits && explicitInits->find(field) != explicitInits->end();
      uint64_t fieldSize = field->type->size();

      const bool isBitField = field->isBitFieldDeclaration() != nullptr;
      if (isBitField) {
        const auto group = BitFieldGroup::startingFrom(
            i, ad->fields.length, [ad](size_t i) { return ad->fields[i]; });

        if (!haveExplicitInit && explicitInits) {
          haveExplicitInit = llvm::any_of(
              group.bitFields, [explicitInits](BitFieldDeclaration *bf) {
                return explicitInits->find(bf) != explicitInits->end();
              });
        }

        fieldSize = group.sizeInBytes;

        // final pass: create unconditional aliases/extra members for the other
        // bit fields
        if (pass == 1) {
          for (size_t j = 1; j < group.bitFields.size(); ++j) {
            auto bf = group.bitFields[j];
            if (bf->offset == group.byteOffset) {
              aliasPairs.push_back({bf, field});
            } else {
              extraBitFieldMembers.push_back(
                  {bf, field->isBitFieldDeclaration()});
            }
          }
        }

        // skip the other bit fields in this pass
        i += group.bitFields.size() - 1; 
      }

      // 1st pass: only for fields with explicit initializer
      if (pass == 0 && !haveExplicitInit)
        continue;

      // final pass: only for fields without explicit initializer
      if (pass == 1 && haveExplicitInit)
        continue;

      // skip empty fields
      if (fieldSize == 0)
        continue;

      const uint64_t f_begin = field->offset;
      const uint64_t f_end = f_begin + fieldSize;

      const auto llType =
          isBitField ? LLIntegerType::get(gIR->context(), fieldSize * 8)
                     : DtoMemType(field->type);

      // check for overlap with existing fields (on a byte level, not bits)
      bool overlaps = false;
      if (isBitField || field->overlapped()) {
        for (const auto &existing : actualFields) {
          const uint64_t e_begin = existing.field->offset;
          const uint64_t e_end = e_begin + existing.size;

          if (e_begin < f_end && e_end > f_begin) {
            overlaps = true;
            if (aliases == Aliases::AddToVarGEPIndices && e_begin == f_begin &&
                existing.llType == llType) {
              aliasPairs.push_back(std::make_pair(field, existing.field));
            }
            break;
          }
        }
      }

      if (!overlaps)
        actualFields.push_back({field, llType, fieldSize});
    }
  }

  // Now we can build a list of LLVM types for the actual LL fields.
  // Make sure to zero out any padding and set the GEP indices for the directly
  // indexable variables.

  // first we sort the list by offset
  std::sort(actualFields.begin(), actualFields.end(),
            [](const Data &l, const Data &r) {
              return l.field->offset < r.field->offset;
            });

  for (const auto &af : actualFields) {
    const auto vd = af.field;
    const auto llType = af.llType;

    if (vd->offset < m_offset) {
      error(vd->loc,
            "%s `%s` overlaps previous field. This is an ICE, please file an "
            "LDC issue.",
            vd->kind(), vd->toPrettyChars());
      fatal();
    }

    // Add an explicit field for any padding so we can zero it, as per TDPL
    // §7.1.1.
    if (m_offset < vd->offset) {
      m_fieldIndex += add_zeros(m_defaultTypes, m_offset, vd->offset);
      m_offset = vd->offset;
    }

    // add default type
    m_defaultTypes.push_back(llType);

    unsigned fieldAlignment, fieldSize;
    if (!llType->isSized()) {
      // forward reference in a cycle or similar, we need to trust the D type
      fieldAlignment = DtoAlignment(vd->type);
      fieldSize = af.size;
    } else {
      fieldAlignment = getABITypeAlign(llType);
      if (vd->isBitFieldDeclaration()) {
        fieldSize = af.size; // an IR integer of possibly non-power-of-2 size
      } else {
        fieldSize = getTypeAllocSize(llType);
        assert(fieldSize <= af.size);
      }
    }

    // advance offset to right past this field
    if (!m_packed) {
      assert(fieldAlignment);
      m_packed = ((m_offset & (fieldAlignment - 1)) != 0);
    }
    m_offset += fieldSize;

    // set the field index
    m_varGEPIndices[vd] = m_fieldIndex;

    // let any aliases reuse this field/GEP index
    for (const auto &pair : aliasPairs) {
      if (pair.second == vd)
        m_varGEPIndices[pair.first] = m_fieldIndex;
    }

    // store extra bit field members of this group
    for (const auto &pair : extraBitFieldMembers) {
      if (pair.second == vd)
        m_extraBitFieldMembers.push_back(pair);
    }

    ++m_fieldIndex;

    m_maxFieldIRAlignment = std::max(m_maxFieldIRAlignment, fieldAlignment);
  }
}

void AggrTypeBuilder::alignCurrentOffset(unsigned alignment) {
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

  // check if the aggregate size makes it packed in IR terms
  if (!m_packed && (aggregateSize & (m_maxFieldIRAlignment - 1)))
    m_packed = true;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration *ad)
    : IrType(ad->type,
             LLStructType::create(gIR->context(), ad->toPrettyChars())),
      aggr(ad) {}

unsigned IrTypeAggr::getMemberLocation(VarDeclaration *var, bool& isFieldIdx) const {
  // Note: The interface is a bit more general than what we actually return.
  // Specifically, the frontend offset information we use for overlapping
  // fields is always based at the object start.
  auto it = varGEPIndices.find(var);
  if (it != varGEPIndices.end()) {
    isFieldIdx = true;
    return it->second;
  } else {
    isFieldIdx = false;
    return var->offset;
  }
}

//////////////////////////////////////////////////////////////////////////////

BitFieldGroup BitFieldGroup::startingFrom(
    size_t startFieldIndex, size_t numTotalFields,
    std::function<VarDeclaration *(size_t i)> getFieldFn) {
  BitFieldGroup group;

  for (size_t i = startFieldIndex; i < numTotalFields; ++i) {
    auto bf = getFieldFn(i)->isBitFieldDeclaration();
    if (!bf)
      break;

    unsigned bitOffset = bf->bitOffset;
    if (i == startFieldIndex) {
      group.byteOffset = bf->offset;
    } else if (bf->offset >= group.byteOffset + group.sizeInBytes ||
               bf->offset < group.byteOffset) { // unions
      // starts a new bit field group
      break;
    } else {
      // the byte offset might not match the group's
      bitOffset += (bf->offset - group.byteOffset) * 8;
    }

    const auto sizeInBytes = (bitOffset + bf->fieldWidth + 7) / 8;
    group.sizeInBytes = std::max(group.sizeInBytes, sizeInBytes);

    group.bitFields.push_back(bf);
  }

  return group;
}

unsigned BitFieldGroup::getBitOffset(BitFieldDeclaration *member) const {
  assert(member->offset >= byteOffset);
  return (member->offset - byteOffset) * 8 + member->bitOffset;
}
