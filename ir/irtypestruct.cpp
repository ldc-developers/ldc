#include "llvm/DerivedTypes.h"

#include "aggregate.h"
#include "declaration.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/utils.h"
#include "ir/irtypestruct.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration * ad)
:   IrType(ad->type, llvm::OpaqueType::get()),
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
            defaultTypes.push_back(llvm::Type::Int64Ty);
            diff -= 8;
        }
        else if (diff % 4 == 0)
        {
            defaultTypes.push_back(llvm::Type::Int32Ty);
            diff -= 4;
        }
        else if (diff % 2 == 0)
        {
            defaultTypes.push_back(llvm::Type::Int16Ty);
            diff -= 2;
        }
        else
        {
            defaultTypes.push_back(llvm::Type::Int8Ty);
            diff -= 1;
        }
    }
    return defaultTypes.size() - n;
}

const llvm::Type* IrTypeStruct::buildType()
{
    IF_LOG Logger::println("Building struct type %s @ %s", sd->toPrettyChars(), sd->locToChars());
    LOG_SCOPE;

    // if it's a forward declaration, all bets are off, stick with the opaque
    if (sd->sizeok != 1)
        return pa.get();

    // find the fields that contribute to the default initializer.
    // these will define the default type.

    std::vector<const llvm::Type*> defaultTypes;
    defaultTypes.reserve(16);

    size_t offset = 0;
    size_t field_index = 0;

    bool packed = (sd->type->alignsize() == 1);

    ArrayIter<VarDeclaration> it(sd->fields);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        //Logger::println("vd: %s", vd->toPrettyChars());

        //assert(vd->ir.irField == NULL && "struct inheritance is not allowed, how can this happen?");

        // skip if offset moved backwards
        if (vd->offset < offset)
        {
            IF_LOG Logger::println("Skipping field %s %s (+%u) for default", vd->type->toChars(), vd->toChars(), vd->offset);
            if (vd->ir.irField == NULL)
                new IrField(vd, 0, vd->offset);
            continue;
        }

        IF_LOG Logger::println("Adding default field %s %s (+%u)", vd->type->toChars(), vd->toChars(), vd->offset);

        // get next aligned offset for this type
        size_t alignedoffset = offset;
        if (!packed)
        {
            size_t alignsize = vd->type->alignsize();
            alignedoffset = (offset + alignsize - 1) & ~(alignsize - 1);
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

        // give field index
        // the IrField creation doesn't really belong here, but it's a trivial operation
        // and it save yet another of these loops.
        IF_LOG Logger::println("Field index: %zu", field_index);
        if (vd->ir.irField == NULL)
            new IrField(vd, field_index);
        field_index++;
    }

    // tail padding?
    if (offset < sd->structsize)
    {
        add_zeros(defaultTypes, sd->structsize - offset);
    }

    // build the llvm type
    const llvm::Type* st = llvm::StructType::get(defaultTypes, packed);

    // refine type
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(st);

    // name types
    Type::sir->getState()->module->addTypeName(sd->toPrettyChars(), pa.get());

#if 0
    IF_LOG Logger::cout() << "final struct type: " << *pa.get() << std::endl;
#endif

    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////
