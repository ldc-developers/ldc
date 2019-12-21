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
  if (v1 && v2) {
    return v1->offset < v2->offset;
  }
  // sort NULL pointers towards the end
  return v1 && !v2;
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

namespace {
enum FieldPriority {
  FP_ExplicitVoid = 0, // lowest priority: fields with explicit void initializer
  FP_Default = 1,      // default initializer
  FP_Explicit = 2,     // explicit non-void initializer
  FP_Value = 3,        // highest priority: values (for literals)
};

FieldPriority prioritize(VarDeclaration *field,
                         const AggrTypeBuilder::VarInitMap *explicitInits) {
  if (explicitInits && explicitInits->find(field) != explicitInits->end())
    return FP_Value;
  if (auto init = field->_init)
    return !init->isVoidInitializer() ? FP_Explicit : FP_ExplicitVoid;
  return FP_Default;
}
}

void AggrTypeBuilder::addAggregate(
    AggregateDeclaration *ad, const AggrTypeBuilder::VarInitMap *explicitInits,
    AggrTypeBuilder::Aliases aliases) {
  const size_t n = ad->fields.length;
  if (n == 0)
    return;

  // prioritize overlapping fields
  LLSmallVector<FieldPriority, 16> priorities;
  priorities.reserve(n);
  for (auto f : ad->fields) {
    priorities.push_back(prioritize(f, explicitInits));
    IF_LOG Logger::println("Field priority for %s: %d", f->toChars(),
                           priorities.back());
  }

  // mirror the ad->fields array but only fill in contributors
  LLSmallVector<VarDeclaration *, 16> data(n, nullptr);

  // list of pairs: alias => actual field (same offset, same LL type)
  LLSmallVector<std::pair<VarDeclaration *, VarDeclaration *>, 16> aliasPairs;

  // one pass per priority in descending order
  const auto minMaxPriority =
      std::minmax_element(priorities.begin(), priorities.end());
  for (int p = *minMaxPriority.second; p >= *minMaxPriority.first; p--) {
    // iterate over fields of that priority, in declaration order
    for (size_t index = 0; index < n; ++index) {
      if (priorities[index] != p)
        continue;

      VarDeclaration *field = ad->fields[index];
      const size_t f_begin = field->offset;
      const size_t f_end = f_begin + field->type->size();

      // skip empty fields
      if (f_begin == f_end)
        continue;

      // check for overlapping existing fields
      bool overlaps = false;
      if (field->overlapped) {
        for (const auto vd : data) {
          if (!vd)
            continue;

          const size_t v_begin = vd->offset;
          const size_t v_end = v_begin + vd->type->size();

          if (v_begin < f_end && v_end > f_begin) {
            if (aliases == Aliases::AddToVarGEPIndices && v_begin == f_begin &&
                DtoMemType(vd->type) == DtoMemType(field->type)) {
              aliasPairs.push_back(std::make_pair(field, vd));
            }
            overlaps = true;
            break;
          }
        }
      }

      if (!overlaps)
        data[index] = field;
    }
  }

  // Now we can build a list of LLVM types for the actual LL fields.
  // Make sure to zero out any padding and set the GEP indices for the directly
  // indexable variables.

  // first we sort the list by offset
  std::sort(data.begin(), data.end(), var_offset_sort_cb);

  for (const auto vd : data) {
    if (!vd)
      continue;

    assert(vd->offset >= m_offset && "Variable overlaps previous field.");

    // Add an explicit field for any padding so we can zero it, as per TDPL
    // §7.1.1.
    if (m_offset < vd->offset) {
      m_fieldIndex += add_zeros(m_defaultTypes, m_offset, vd->offset);
      m_offset = vd->offset;
    }

    // add default type
    m_defaultTypes.push_back(DtoMemType(vd->type));

    // advance offset to right past this field
    m_offset += getMemberSize(vd->type);

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
  if (ad->sizeok == SIZEOKdone) {
    aggregateSize = ad->structsize;

    const auto naturalAlignment = ad->alignsize;
    auto explicitAlignment = STRUCTALIGN_DEFAULT;
    if (auto sd = ad->isStructDeclaration())
      explicitAlignment = sd->alignment;

    aggregateAlignment = explicitAlignment == STRUCTALIGN_DEFAULT
                             ? naturalAlignment
                             : explicitAlignment;
  }

  // Classes apparently aren't padded; their size may not match the alignment.
  assert((ad->isClassDeclaration() ||
          (aggregateSize & (aggregateAlignment - 1)) == 0) &&
         "Size not a multiple of alignment?");

  // For unions, only a subset of the fields are actually used for the IR type -
  // don't care (about a few potentially needlessly packed IR structs).
  for (const auto field : ad->fields) {
    const auto naturalFieldTypeAlignment = field->type->alignsize();
    const auto explicitFieldTypeAlignment = field->type->alignment();
    const auto fieldTypeAlignment =
        explicitFieldTypeAlignment == STRUCTALIGN_DEFAULT
            ? naturalFieldTypeAlignment
            : explicitFieldTypeAlignment;

    // The aggregate size, aggregate alignment and the field offset need to be
    // multiples of the field type's alignment, otherwise the aggregate type is
    // unnaturally aligned, and LLVM would insert padding.
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
