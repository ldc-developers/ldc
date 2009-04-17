#include "llvm/DerivedTypes.h"

#include "aggregate.h"
#include "declaration.h"
#include "dsymbol.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////

extern size_t add_zeros(std::vector<const llvm::Type*>& defaultTypes, size_t diff);

//////////////////////////////////////////////////////////////////////////////

IrTypeClass::IrTypeClass(ClassDeclaration* cd)
:   IrTypeAggr(cd),
    cd(cd),
    tc((TypeClass*)cd->type),
    vtbl_pa(llvm::OpaqueType::get())
{
    vtbl_size = cd->vtbl.dim;
    num_interface_vtbls = 0;
}

//////////////////////////////////////////////////////////////////////////////

void IrTypeClass::addBaseClassData(
    std::vector< const llvm::Type * > & defaultTypes,
    ClassDeclaration * base,
    size_t & offset,
    size_t & field_index)
{
    if (base->baseClass)
    {
        addBaseClassData(defaultTypes, base->baseClass, offset, field_index);
    }

    ArrayIter<VarDeclaration> it(base->fields);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();

        // skip if offset moved backwards
        if (vd->offset < offset)
        {
            IF_LOG Logger::println("Skipping field %s %s (+%u) for default", vd->type->toChars(), vd->toChars(), vd->offset);
            if (vd->ir.irField == NULL)
            {
                new IrField(vd, 2, vd->offset - PTRSIZE * 2);
            }
            continue;
        }

        IF_LOG Logger::println("Adding default field %s %s (+%u)", vd->type->toChars(), vd->toChars(), vd->offset);

        // get next aligned offset for this type
        size_t alignsize = vd->type->alignsize();
        size_t alignedoffset = (offset + alignsize - 1) & ~(alignsize - 1);

        // do we need to insert explicit padding before the field?
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
        {
            new IrField(vd, field_index);
        }
        field_index++;
    }

    // any interface implementations?
    if (base->vtblInterfaces)
    {
        bool new_instances = (base == cd);

        ArrayIter<BaseClass> it2(*base->vtblInterfaces);

        VarDeclarationIter interfaces_idx(ClassDeclaration::classinfo->fields, 3);
        Type* first = interfaces_idx->type->next->pointerTo();

        for (; !it2.done(); it2.next())
        {
            BaseClass* b = it2.get();
            IF_LOG Logger::println("Adding interface vtbl for %s", b->base->toPrettyChars());

            Array arr;
            b->fillVtbl(cd, &arr, new_instances);

            const llvm::Type* ivtbl_type = buildVtblType(first, &arr);
            defaultTypes.push_back(llvm::PointerType::get(ivtbl_type, 0));

            offset += PTRSIZE;

            // add to the interface map
            addInterfaceToMap(b->base, field_index);
            field_index++;

            // inc count
            num_interface_vtbls++;
        }
    }

    // tail padding?
    if (offset < base->structsize)
    {
        field_index += add_zeros(defaultTypes, base->structsize - offset);
        offset = base->structsize;
    }
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type* IrTypeClass::buildType()
{
    IF_LOG Logger::println("Building class type %s @ %s", cd->toPrettyChars(), cd->locToChars());
    LOG_SCOPE;
    IF_LOG Logger::println("Instance size: %u", cd->structsize);

    // find the fields that contribute to the default initializer.
    // these will define the default type.

    std::vector<const llvm::Type*> defaultTypes;
    defaultTypes.reserve(32);

    // add vtbl
    defaultTypes.push_back(llvm::PointerType::get(vtbl_pa.get(), 0));

    // interface are just a vtable
    if (!cd->isInterfaceDeclaration())
    {
        // add monitor
        defaultTypes.push_back(llvm::PointerType::get(llvm::Type::Int8Ty, 0));

        // we start right after the vtbl and monitor
        size_t offset = PTRSIZE * 2;
        size_t field_index = 2;

        // add data members recursively
        addBaseClassData(defaultTypes, cd, offset, field_index);
    }

    // errors are fatal during codegen
    if (global.errors)
        fatal();

    // build the llvm type
    const llvm::Type* st = llvm::StructType::get(defaultTypes, false);

    // refine type
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(st);

    // name type
    Type::sir->getState()->module->addTypeName(cd->toPrettyChars(), pa.get());

    // VTBL

    // build vtbl type
    const llvm::Type* vtblty = buildVtblType(
        ClassDeclaration::classinfo->type,
        &cd->vtbl);

    // refine vtbl pa
    llvm::cast<llvm::OpaqueType>(vtbl_pa.get())->refineAbstractTypeTo(vtblty);

    // name vtbl type
    std::string name(cd->toPrettyChars());
    name.append(".__vtbl");
    Type::sir->getState()->module->addTypeName(name, vtbl_pa.get());

#if 0
    IF_LOG Logger::cout() << "class type: " << *pa.get() << std::endl;
#endif

    return get();
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type* IrTypeClass::buildVtblType(Type* first, Array* vtbl_array)
{
    IF_LOG Logger::println("Building vtbl type for class %s", cd->toPrettyChars());
    LOG_SCOPE;

    std::vector<const llvm::Type*> types;
    types.reserve(vtbl_array->dim);

    // first comes the classinfo
    types.push_back(DtoType(first));

    // then come the functions
    ArrayIter<Dsymbol> it(*vtbl_array);
    it.index = 1;

    for (; !it.done(); it.next())
    {
        Dsymbol* dsym = it.get();
        if (dsym == NULL)
        {
            // FIXME
            // why is this null?
            // happens for mini/s.d
            types.push_back(getVoidPtrType());
            continue;
        }

        FuncDeclaration* fd = dsym->isFuncDeclaration();
        assert(fd && "invalid vtbl entry");

        IF_LOG Logger::println("Adding type of %s", fd->toPrettyChars());

        types.push_back(DtoType(fd->type->pointerTo()));
    }

    // build the vtbl llvm type
    return llvm::StructType::get(types, false);
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeClass::get()
{
    return llvm::PointerType::get(pa.get(), 0);
}

//////////////////////////////////////////////////////////////////////////////

size_t IrTypeClass::getInterfaceIndex(ClassDeclaration * inter)
{
    ClassIndexMap::iterator it = interfaceMap.find(inter);
    if (it == interfaceMap.end())
        return ~0;
    return it->second;
}

//////////////////////////////////////////////////////////////////////////////

void IrTypeClass::addInterfaceToMap(ClassDeclaration * inter, size_t index)
{
    // don't duplicate work or overwrite indices
    if (interfaceMap.find(inter) != interfaceMap.end())
        return;

    // add this interface
    interfaceMap.insert(std::make_pair(inter, index));

    // add all its base interfaces recursively
    for (size_t i = 0; i < inter->interfaces_dim; i++)
    {
        BaseClass* b = inter->interfaces[i];
        addInterfaceToMap(b->base, index);
    }
}

//////////////////////////////////////////////////////////////////////////////
