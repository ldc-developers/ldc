//===-- irtypeclass.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 303
#include "llvm/IR/DerivedTypes.h"
#else
#include "llvm/DerivedTypes.h"
#endif

#include "aggregate.h"
#include "declaration.h"
#include "dsymbol.h"
#include "mtype.h"
#include "target.h"
#include "template.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/functions.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////

extern size_t add_zeros(std::vector<llvm::Type*>& defaultTypes,
    size_t startOffset, size_t endOffset);
extern bool var_offset_sort_cb(const VarDeclaration* v1, const VarDeclaration* v2);

//////////////////////////////////////////////////////////////////////////////

IrTypeClass::IrTypeClass(ClassDeclaration* cd)
:   IrTypeAggr(cd),
    cd(cd),
    tc(static_cast<TypeClass*>(cd->type))
{
    std::string vtbl_name(cd->toPrettyChars());
    vtbl_name.append(".__vtbl");
    vtbl_type = LLStructType::create(gIR->context(), vtbl_name);
    vtbl_size = cd->vtbl.dim;
    num_interface_vtbls = 0;
}

//////////////////////////////////////////////////////////////////////////////

void IrTypeClass::addBaseClassData(
    std::vector<llvm::Type *> & defaultTypes,
    ClassDeclaration * base,
    size_t & offset,
    size_t & field_index)
{
    if (base->baseClass)
    {
        addBaseClassData(defaultTypes, base->baseClass, offset, field_index);
    }

    // FIXME: merge code with structs in IrTypeAggr

    // mirror the sd->fields array but only fill in contributors
    const size_t n = base->fields.dim;
    LLSmallVector<VarDeclaration*, 16> data(n, NULL);
    default_fields.reserve(n);

    // first fill in the fields with explicit initializers
    for (size_t index = 0; index < n; ++index)
    {
        VarDeclaration *field = base->fields[index];

        // init is !null for explicit inits
        if (field->init != NULL)
        {
            IF_LOG Logger::println("adding explicit initializer for struct field %s",
                field->toChars());

            data[index] = field;

            size_t f_begin = field->offset;
            size_t f_end = f_begin + field->type->size();

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

                    base->error(vd->loc, "has overlapping initialization for %s and %s",
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
        VarDeclaration *field = base->fields[index];

        size_t f_begin = field->offset;
        size_t f_end = f_begin + field->type->size();

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

    // first we sort the list by offset
    std::sort(data.begin(), data.end(), var_offset_sort_cb);

    // add types to list
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = data[i];

        if (vd == NULL)
            continue;

        assert(vd->offset >= offset && "it's a bug... most likely DMD bug 2481");

        // add to default field list
        if (cd == base)
            default_fields.push_back(vd);

        // get next aligned offset for this type
        size_t alignedoffset = realignOffset(offset, vd->type);

        // insert explicit padding?
        if (alignedoffset < vd->offset)
        {
            field_index += add_zeros(defaultTypes, alignedoffset, vd->offset);
        }

        // add default type
        defaultTypes.push_back(DtoType(vd->type)); // @@@ i1ToI8?!

        // advance offset to right past this field
        offset = vd->offset + vd->type->size();

        // set the field index
        getIrField(vd, true)->setAggrIndex(static_cast<unsigned>(field_index));
        ++field_index;
    }

    // any interface implementations?
    if (base->vtblInterfaces && base->vtblInterfaces->dim > 0)
    {
        bool new_instances = (base == cd);

        VarDeclaration *interfaces_idx = Type::typeinfoclass->fields[3];
        Type* first = interfaces_idx->type->nextOf()->pointerTo();

        // align offset
        offset = (offset + Target::ptrsize - 1) & ~(Target::ptrsize - 1);

        for (BaseClasses::iterator I = base->vtblInterfaces->begin(),
                                   E = base->vtblInterfaces->end();
                                   I != E; ++I)
        {
            BaseClass *b = *I;
            IF_LOG Logger::println("Adding interface vtbl for %s", b->base->toPrettyChars());

            FuncDeclarations arr;
            b->fillVtbl(cd, &arr, new_instances);

            llvm::Type* ivtbl_type = llvm::StructType::get(gIR->context(), buildVtblType(first, &arr));
            defaultTypes.push_back(llvm::PointerType::get(ivtbl_type, 0));

            offset += Target::ptrsize;

            // add to the interface map
            addInterfaceToMap(b->base, field_index);
            field_index++;

            // inc count
            num_interface_vtbls++;
        }
    }

#if 0
    // tail padding?
    if (offset < base->structsize)
    {
        field_index += add_zeros(defaultTypes, offset, base->structsize);
        offset = base->structsize;
    }
#endif
}

//////////////////////////////////////////////////////////////////////////////

IrTypeClass* IrTypeClass::get(ClassDeclaration* cd)
{
    IrTypeClass* t = new IrTypeClass(cd);
    cd->type->irtype = t;

    IF_LOG Logger::println("Building class type %s @ %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;
    IF_LOG Logger::println("Instance size: %u", cd->structsize);

    // find the fields that contribute to the default initializer.
    // these will define the default type.

    std::vector<llvm::Type*> defaultTypes;
    defaultTypes.reserve(32);

    // add vtbl
    defaultTypes.push_back(llvm::PointerType::get(t->vtbl_type, 0));

    // interfaces are just a vtable
    if (cd->isInterfaceDeclaration())
    {
        t->num_interface_vtbls = cd->vtblInterfaces ? cd->vtblInterfaces->dim : 0;
    }
    // classes have monitor and fields
    else
    {
        size_t offset;
        size_t field_index;
        if (!cd->isCPPclass() && !cd->isCPPinterface()) {
            // add monitor
            defaultTypes.push_back(llvm::PointerType::get(llvm::Type::getInt8Ty(gIR->context()), 0));

            // we start right after the vtbl and monitor
            offset = Target::ptrsize * 2;
            field_index = 2;
        } else {
            // C++ classes does not have a monitor
            offset = Target::ptrsize;
            field_index = 1;
        }

        // add data members recursively
        t->addBaseClassData(defaultTypes, cd, offset, field_index);

#if 1
        // tail padding?
        if (offset < cd->structsize)
        {
            field_index += add_zeros(defaultTypes, offset, cd->structsize);
            offset = cd->structsize;
        }
#endif
    }

    // errors are fatal during codegen
    if (global.errors)
        fatal();

    // set struct body
    isaStruct(t->type)->setBody(defaultTypes, false);

    // VTBL

    // set vtbl type body
    FuncDeclarations vtbl;
    vtbl.reserve(cd->vtbl.dim);
    vtbl.push(0);
    for (size_t i = cd->vtblOffset(); i < cd->vtbl.dim; ++i)
    {
        FuncDeclaration *fd = cd->vtbl[i]->isFuncDeclaration();
        assert(fd);
        vtbl.push(fd);
    }
    t->vtbl_type->setBody(t->buildVtblType(Type::typeinfoclass->type, &vtbl));

    IF_LOG Logger::cout() << "class type: " << *t->type << std::endl;

    return t;
}

//////////////////////////////////////////////////////////////////////////////

std::vector<llvm::Type*> IrTypeClass::buildVtblType(Type* first, FuncDeclarations* vtbl_array)
{
    IF_LOG Logger::println("Building vtbl type for class %s", cd->toPrettyChars());
    LOG_SCOPE;

    std::vector<llvm::Type*> types;
    types.reserve(vtbl_array->dim);

    // first comes the classinfo
    if (!cd->isCPPclass() && !cd->isCPPinterface())
        types.push_back(DtoType(first));

    // then come the functions
    for (FuncDeclarations::iterator I = vtbl_array->begin() + 1,
                                    E = vtbl_array->end();
                                    I != E; ++I)
    {
        FuncDeclaration* fd = *I;
        if (fd == NULL)
        {
            // FIXME
            // why is this null?
            // happens for mini/s.d
            types.push_back(getVoidPtrType());
            continue;
        }

        IF_LOG Logger::println("Adding type of %s", fd->toPrettyChars());

        // If inferring return type and semantic3 has not been run, do it now.
        // This pops up in some other places in the frontend as well, however
        // it is probably a bug that it still occurs that late.
        if (!fd->type->nextOf() && fd->inferRetType)
        {
            Logger::println("Running late semantic3 to infer return type.");
            TemplateInstance *spec = fd->isSpeculative();
            unsigned int olderrs = global.errors;
            fd->semantic3(fd->scope);
            if (spec && global.errors != olderrs)
                spec->errors = global.errors - olderrs;
        }

        if (!fd->type->nextOf()) {
            // Return type of the function has not been inferred. This seems to
            // happen with virtual functions and is probably a frontend bug.
            IF_LOG Logger::println("Broken function type, semanticRun: %d",
                fd->semanticRun);
            types.push_back(getVoidPtrType());
            continue;
        }

        types.push_back(getPtrToType(DtoFunctionType(fd)));

    }

    return types;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeClass::getLLType()
{
    return llvm::PointerType::get(type, 0);
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeClass::getMemoryLLType()
{
    return type;
}

//////////////////////////////////////////////////////////////////////////////

size_t IrTypeClass::getInterfaceIndex(ClassDeclaration * inter)
{
    ClassIndexMap::iterator it = interfaceMap.find(inter);
    if (it == interfaceMap.end())
        return ~0UL;
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

    // add the direct base interfaces recursively - they
    // are accessed through the same index
    if (inter->interfaces_dim > 0)
    {
        BaseClass* b = inter->interfaces[0];
        addInterfaceToMap(b->base, index);
    }
}

//////////////////////////////////////////////////////////////////////////////
