//===-- irtypestruct.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irtypestruct.h"

#if LDC_LLVM_VER >= 303
#include "llvm/IR/DerivedTypes.h"
#else
#include "llvm/DerivedTypes.h"
#endif

#include "aggregate.h"
#include "declaration.h"
#include "init.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeStruct::IrTypeStruct(StructDeclaration * sd)
:   IrTypeAggr(sd),
    sd(sd),
    ts(static_cast<TypeStruct*>(sd->type))
{
}

//////////////////////////////////////////////////////////////////////////////

static bool isAligned(llvm::Type* type, size_t offset) {
    return gDataLayout->getABITypeAlignment(type) % offset == 0;
}

size_t add_zeros(std::vector<llvm::Type*>& defaultTypes,
    size_t startOffset, size_t endOffset)
{
    size_t const oldLength = defaultTypes.size();

    llvm::Type* const eightByte = llvm::Type::getInt64Ty(gIR->context());
    llvm::Type* const fourByte = llvm::Type::getInt32Ty(gIR->context());
    llvm::Type* const twoByte = llvm::Type::getInt16Ty(gIR->context());

    assert(startOffset <= endOffset);
    size_t paddingLeft = endOffset - startOffset;
    while (paddingLeft)
    {
        if (global.params.is64bit && paddingLeft >= 8 && isAligned(eightByte, startOffset))
        {
            defaultTypes.push_back(eightByte);
            startOffset += 8;
        }
        else if (paddingLeft >= 4 && isAligned(fourByte, startOffset))
        {
            defaultTypes.push_back(fourByte);
            startOffset += 4;
        }
        else if (paddingLeft >= 2 && isAligned(twoByte, startOffset))
        {
            defaultTypes.push_back(twoByte);
            startOffset += 2;
        }
        else
        {
            defaultTypes.push_back(llvm::Type::getInt8Ty(gIR->context()));
            startOffset += 1;
        }

        paddingLeft = endOffset - startOffset;
    }

    return defaultTypes.size() - oldLength;
}


bool var_offset_sort_cb(const VarDeclaration* v1, const VarDeclaration* v2)
{
    if (v1 && v2)
        return v1->offset < v2->offset;
    // sort NULL pointers towards the end
    return v1 && !v2;
}

// this is pretty much the exact same thing we need to do for fields in each
// base class of a class

IrTypeStruct* IrTypeStruct::get(StructDeclaration* sd)
{
    IrTypeStruct* t = new IrTypeStruct(sd);
    sd->type->ctype = t;

    IF_LOG Logger::println("Building struct type %s @ %s",
        sd->toPrettyChars(), sd->loc.toChars());
    LOG_SCOPE;

    // if it's a forward declaration, all bets are off, stick with the opaque
    if (sd->sizeok != SIZEOKdone)
        return t;

    // TODO:: Somehow merge this with IrAggr::createInitializerConstant, or
    // replace it by just taking the type of the default initializer.

    // mirror the sd->fields array but only fill in contributors
    const size_t n = sd->fields.dim;
    LLSmallVector<VarDeclaration*, 16> data(n, NULL);

    // first fill in the fields with explicit initializers
    for (size_t index = 0; index < n; ++index)
    {
        VarDeclaration *field = sd->fields[index];

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

                    sd->error(vd->loc, "has overlapping initialization for %s and %s",
                        field->toChars(), vd->toChars());
                }
            }
        }
    }

    if (global.errors)
    {
        fatal();
    }

    // fill in default initializers
    for (size_t index = 0; index < n; ++index)
    {
        if (data[index])
            continue;
        VarDeclaration *field = sd->fields[index];

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

    // ok. now we can build a list of llvm types. and make sure zeros are inserted if necessary.
    std::vector<LLType*> defaultTypes;
    defaultTypes.reserve(16);

    size_t offset = 0;
    size_t field_index = 0;

    bool packed = (sd->alignment == 1);

    // first we sort the list by offset
    std::sort(data.begin(), data.end(), var_offset_sort_cb);

    // add types to list
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = data[i];

        if (vd == NULL)
            continue;

        assert(vd->offset >= offset);

        // get next aligned offset for this type
        size_t alignedoffset = offset;
        if (!packed)
        {
            alignedoffset = realignOffset(alignedoffset, vd->type);
        }

        // insert explicit padding?
        if (alignedoffset < vd->offset)
        {
            field_index += add_zeros(defaultTypes, alignedoffset, vd->offset);
        }

        // add default type
        defaultTypes.push_back(DtoType(vd->type));

        // advance offset to right past this field
        offset = vd->offset + vd->type->size();

        // set the field index
        getIrField(vd, true)->setAggrIndex(static_cast<unsigned>(field_index));
        ++field_index;
    }

    // tail padding?
    if (offset < sd->structsize)
    {
        add_zeros(defaultTypes, offset, sd->structsize);
    }

    // set struct body
    isaStruct(t->type)->setBody(defaultTypes, packed);

    IF_LOG Logger::cout() << "final struct type: " << *t->type << std::endl;

    return t;
}
