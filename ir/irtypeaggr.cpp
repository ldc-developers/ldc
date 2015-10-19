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
#include "init.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// FIXME A similar function is in ir/iraggr.cpp and RTTIBuilder::push().
static inline
size_t add_zeros(std::vector<llvm::Type*>& defaultTypes,
    size_t startOffset, size_t endOffset)
{
    assert(startOffset <= endOffset);
    const size_t paddingSize = endOffset - startOffset;
    if (paddingSize)
    {
        llvm::ArrayType* pad = llvm::ArrayType::get(llvm::Type::getInt8Ty(gIR->context()), paddingSize);
        defaultTypes.push_back(pad);
    }
    return paddingSize ? 1 : 0;
}


bool var_offset_sort_cb(const VarDeclaration* v1, const VarDeclaration* v2)
{
    if (v1 && v2)
        return v1->offset < v2->offset;
    // sort NULL pointers towards the end
    return v1 && !v2;
}

AggrTypeBuilder::AggrTypeBuilder(bool packed) :
    m_offset(0), m_fieldIndex(0), m_overallAlignment(0), m_packed(packed)
{
    m_defaultTypes.reserve(32);
}

void AggrTypeBuilder::addType(llvm::Type *type, unsigned size)
{
    m_defaultTypes.push_back(type);
    m_offset += size;
    m_fieldIndex++;
}

void AggrTypeBuilder::addAggregate(AggregateDeclaration *ad)
{
    // mirror the ad->fields array but only fill in contributors
    const size_t n = ad->fields.dim;
    LLSmallVector<VarDeclaration*, 16> data(n, NULL);

    unsigned int errors = global.errors;

    // first fill in the fields with explicit initializers
    for (size_t index = 0; index < n; ++index)
    {
        VarDeclaration *field = ad->fields[index];

        // init is !null for explicit inits
        if (field->init != NULL && !field->init->isVoidInitializer())
        {
            IF_LOG Logger::println("adding explicit initializer for struct field %s",
                field->toChars());

            size_t f_size = field->type->size();
            size_t f_begin = field->offset;
            size_t f_end = f_begin + f_size;
            if (f_size == 0)
                continue;

            data[index] = field;

            // make sure there is no overlap
            for (size_t i = 0; i < index; i++)
            {
                if (data[i] != NULL)
                {
                    VarDeclaration* vd = data[i];
                    size_t v_begin = vd->offset;
                    size_t v_end = v_begin + vd->type->size();

                    if (v_begin >= f_end || v_end <= f_begin)
                        continue;

                    ad->error(vd->loc, "has overlapping initialization for %s and %s",
                        field->toChars(), vd->toChars());
                }
            }
        }
    }

    if (errors != global.errors)
    {
        // There was an overlapping initialization.
        // Return if errors are gagged otherwise abort.
        if (global.gag) return;
        fatal();
    }

    // fill in default initializers
    for (size_t index = 0; index < n; ++index)
    {
        if (data[index])
            continue;
        VarDeclaration *field = ad->fields[index];

        size_t f_size = field->type->size();
        size_t f_begin = field->offset;
        size_t f_end = f_begin + f_size;
        if (f_size == 0)
            continue;

        // make sure it doesn't overlap anything explicit
        bool overlaps = false;
        for (size_t i = 0; i < n; i++)
        {
            if (data[i])
            {
                size_t v_begin = data[i]->offset;
                size_t v_end = v_begin + data[i]->type->size();

                if (v_begin >= f_end || v_end <= f_begin)
                    continue;

                overlaps = true;
                break;
            }
        }

        // if no overlap was found, add the default initializer
        if (!overlaps)
        {
            IF_LOG Logger::println("adding default initializer for struct field %s",
                field->toChars());
            data[index] = field;
        }
    }

    //
    // ok. now we can build a list of llvm types. and make sure zeros are inserted if necessary.
    //

    // first we sort the list by offset
    std::sort(data.begin(), data.end(), var_offset_sort_cb);

    // add types to list
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = data[i];

        if (vd == NULL)
            continue;

        assert(vd->offset >= m_offset && "Variable overlaps previous field.");

        // Add an explicit field for any padding so we can zero it, as per TDPL §7.1.1.
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
        ++m_fieldIndex;
    }
}

void AggrTypeBuilder::alignCurrentOffset(unsigned alignment)
{
    m_overallAlignment = std::max(alignment, m_overallAlignment);

    unsigned aligned = (m_offset + alignment - 1) & ~(alignment - 1);
    if (m_offset < aligned) {
        m_fieldIndex += add_zeros(m_defaultTypes, m_offset, aligned);
        m_offset = aligned;
    }
}

void AggrTypeBuilder::addTailPadding(unsigned aggregateSize)
{
    // tail padding?
    if (m_offset < aggregateSize)
    {
        add_zeros(m_defaultTypes, m_offset, aggregateSize);
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration * ad)
:   IrType(ad->type, LLStructType::create(gIR->context(), ad->toPrettyChars())),
    diCompositeType(), aggr(ad)
{
}

bool IrTypeAggr::isPacked(AggregateDeclaration* ad)
{
    if (ad->isUnionDeclaration()) return true;
    for (unsigned i = 0; i < ad->fields.dim; i++)
    {
        VarDeclaration* vd = static_cast<VarDeclaration*>(ad->fields.data[i]);
        unsigned a = vd->type->alignsize() - 1;
        if (((vd->offset + a) & ~a) != vd->offset)
            return true;
    }
    return false;
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
