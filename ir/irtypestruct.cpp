#include "llvm/DerivedTypes.h"

#include "aggregate.h"
#include "declaration.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/utils.h"
#include "gen/llvmhelpers.h"
#include "ir/irtypestruct.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration * ad)
:   IrType(ad->type, llvm::OpaqueType::get(gIR->context())),
    aggr(ad)
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeStruct::IrTypeStruct(StructDeclaration * sd)
:   IrTypeAggr(sd),
    sd(sd),
    ts((TypeStruct*)sd->type)
{
}

//////////////////////////////////////////////////////////////////////////////

size_t add_zeros(std::vector<const llvm::Type*>& defaultTypes, size_t diff)
{
    size_t n = defaultTypes.size();
    while (diff)
    {
        if (global.params.is64bit && diff % 8 == 0)
        {
            defaultTypes.push_back(llvm::Type::getInt64Ty(gIR->context()));
            diff -= 8;
        }
        else if (diff % 4 == 0)
        {
            defaultTypes.push_back(llvm::Type::getInt32Ty(gIR->context()));
            diff -= 4;
        }
        else if (diff % 2 == 0)
        {
            defaultTypes.push_back(llvm::Type::getInt16Ty(gIR->context()));
            diff -= 2;
        }
        else
        {
            defaultTypes.push_back(llvm::Type::getInt8Ty(gIR->context()));
            diff -= 1;
        }
    }
    return defaultTypes.size() - n;
}

bool var_offset_sort_cb(const VarDeclaration* v1, const VarDeclaration* v2)
{
    if (v1 && v2)
        return v1->offset < v2->offset;
    else
        return false;
}

// this is pretty much the exact same thing we need to do for fields in each
// base class of a class

const llvm::Type* IrTypeStruct::buildType()
{
    IF_LOG Logger::println("Building struct type %s @ %s",
        sd->toPrettyChars(), sd->loc.toChars());
    LOG_SCOPE;

    // if it's a forward declaration, all bets are off, stick with the opaque
    if (sd->sizeok != 1)
        return pa.get();

    // mirror the sd->fields array but only fill in contributors
    size_t n = sd->fields.dim;
    LLSmallVector<VarDeclaration*, 16> data(n, NULL);
    default_fields.reserve(n);

    // first fill in the fields with explicit initializers
    VarDeclarationIter field_it(sd->fields);
    for (; field_it.more(); field_it.next())
    {
        // init is !null for explicit inits
        if (field_it->init != NULL)
        {
            IF_LOG Logger::println("adding explicit initializer for struct field %s",
                field_it->toChars());

            data[field_it.index] = *field_it;

            size_t f_begin = field_it->offset;
            size_t f_end = f_begin + field_it->type->size();

            // make sure there is no overlap
            for (size_t i = 0; i < field_it.index; i++)
            {
                if (data[i] != NULL)
                {
                    VarDeclaration* vd = data[i];
                    size_t v_begin = vd->offset;
                    size_t v_end = v_begin + vd->type->size();

                    if (v_begin >= f_end || v_end <= f_begin)
                        continue;

                    sd->error(vd->loc, "has overlapping initialization for %s and %s",
                        field_it->toChars(), vd->toChars());
                }
            }
        }
    }

    if (global.errors)
    {
        fatal();
    }

    // fill in default initializers
    field_it = VarDeclarationIter(sd->fields);
    for (;field_it.more(); field_it.next())
    {
        if (data[field_it.index])
            continue;

        size_t f_begin = field_it->offset;
        size_t f_end = f_begin + field_it->type->size();

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
                field_it->toChars());
            data[field_it.index] = *field_it;
        }
    }

    // ok. now we can build a list of llvm types. and make sure zeros are inserted if necessary.
    std::vector<const llvm::Type*> defaultTypes;
    defaultTypes.reserve(16);

    size_t offset = 0;
    size_t field_index = 0;

    bool packed = (sd->type->alignsize() == 1);

    // first we sort the list by offset
    std::sort(data.begin(), data.end(), var_offset_sort_cb);

    // add types to list
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = data[i];

        if (vd == NULL)
            continue;

        assert(vd->offset >= offset);

        // add to default field list
        default_fields.push_back(vd);

        // get next aligned offset for this type
        size_t alignedoffset = offset;
        if (!packed)
        {
            alignedoffset = realignOffset(alignedoffset, vd->type);
        }

        // insert explicit padding?
        if (alignedoffset < vd->offset)
        {
            field_index += add_zeros(defaultTypes, vd->offset - alignedoffset);
        }

        // add default type
        defaultTypes.push_back(DtoType(vd->type));

        // advance offset to right past this field
        offset = vd->offset + vd->type->size();

        // set the field index
        vd->aggrIndex = (unsigned)field_index++;
    }

    // tail padding?
    if (offset < sd->structsize)
    {
        add_zeros(defaultTypes, sd->structsize - offset);
    }

    // build the llvm type
    const llvm::Type* st = llvm::StructType::get(gIR->context(), defaultTypes, packed);

    // refine type
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(st);

    // name types
    Type::sir->getState()->module->addTypeName(sd->toPrettyChars(), pa.get());

    IF_LOG Logger::cout() << "final struct type: " << *pa.get() << std::endl;

    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////
