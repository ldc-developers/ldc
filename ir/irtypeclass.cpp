//===-- irtypeclass.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"

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

void IrTypeClass::addBaseClassData(AggrTypeBuilder &builder, ClassDeclaration *base)
{
    if (base->baseClass)
    {
        addBaseClassData(builder, base->baseClass);
    }

    builder.addAggregate(base);

    // any interface implementations?
    if (base->vtblInterfaces && base->vtblInterfaces->dim > 0)
    {
        bool new_instances = (base == cd);

        VarDeclaration *interfaces_idx = Type::typeinfoclass->fields[3];
        Type* first = interfaces_idx->type->nextOf()->pointerTo();

        // align offset
        builder.alignCurrentOffset(Target::ptrsize);

        for (auto b : *base->vtblInterfaces)
        {
            IF_LOG Logger::println("Adding interface vtbl for %s", b->base->toPrettyChars());

            FuncDeclarations arr;
            b->fillVtbl(cd, &arr, new_instances);

            // add to the interface map
            addInterfaceToMap(b->base, builder.currentFieldIndex());

            llvm::Type* ivtbl_type = llvm::StructType::get(gIR->context(), buildVtblType(first, &arr));
            builder.addType(llvm::PointerType::get(ivtbl_type, 0), Target::ptrsize);

            // inc count
            num_interface_vtbls++;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

IrTypeClass* IrTypeClass::get(ClassDeclaration* cd)
{
    IrTypeClass* t = new IrTypeClass(cd);
    cd->type->ctype = t;

    IF_LOG Logger::println("Building class type %s @ %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;
    IF_LOG Logger::println("Instance size: %u", cd->structsize);

    // This class may contain an align declaration. See issue 726.
    t->packed = false;
    for (ClassDeclaration *base = cd; base != 0 && !t->packed; base = base->baseClass)
    {
        t->packed = isPacked(base);
    }

    AggrTypeBuilder builder(t->packed);

    // add vtbl
    builder.addType(llvm::PointerType::get(t->vtbl_type, 0), Target::ptrsize);

    // interfaces are just a vtable
    if (cd->isInterfaceDeclaration())
    {
        t->num_interface_vtbls = cd->vtblInterfaces ? cd->vtblInterfaces->dim : 0;
    }
    // classes have monitor and fields
    else
    {
        if (!cd->isCPPclass() && !cd->isCPPinterface())
        {
            // add monitor
            builder.addType(llvm::PointerType::get(llvm::Type::getInt8Ty(gIR->context()), 0), Target::ptrsize);
        }

        // add data members recursively
        t->addBaseClassData(builder, cd);

        // add tail padding
        builder.addTailPadding(cd->structsize);
    }

    // errors are fatal during codegen
    if (global.errors)
        fatal();

    // set struct body and copy GEP indices
    isaStruct(t->type)->setBody(builder.defaultTypes(), t->packed);
    t->varGEPIndices = builder.varGEPIndices();

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
    for (auto I = vtbl_array->begin() + 1, E = vtbl_array->end(); I != E; ++I)
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
    auto it = interfaceMap.find(inter);
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
