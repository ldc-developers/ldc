//===-- irstruct.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"
#include "mtype.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "ir/irstruct.h"
#include "ir/irtypeclass.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////

IrStruct::IrStruct(AggregateDeclaration* aggr)
:   diCompositeType(NULL),
    init_type(LLStructType::create(gIR->context(), std::string(aggr->toPrettyChars()) + "_init"))
{
    aggrdecl = aggr;

    type = aggr->type;

    packed = (type->ty == Tstruct)
        ? type->alignsize() == 1
        : false;

    // above still need to be looked at

    init = NULL;
    constInit = NULL;

    vtbl = NULL;
    constVtbl = NULL;
    classInfo = NULL;
    constClassInfo = NULL;

    classInterfacesArray = NULL;
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable * IrStruct::getInitSymbol()
{
    if (init)
        return init;

    // create the initZ symbol
    std::string initname("_D");
    initname.append(aggrdecl->mangle());
    initname.append("6__initZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(aggrdecl);

    init = new llvm::GlobalVariable(
        *gIR->module, init_type, true, _linkage, NULL, initname);

    // set alignment
    init->setAlignment(type->alignsize());
    StructDeclaration *sd = aggrdecl->isStructDeclaration();
    if (sd && sd->alignment != STRUCTALIGN_DEFAULT)
        init->setAlignment(sd->alignment);

    return init;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Constant * IrStruct::getDefaultInit()
{
    if (constInit)
        return constInit;

    std::vector<LLConstant*> constants = type->ty == Tstruct ?
                createStructDefaultInitializer() :
                createClassDefaultInitializer();

    // set initializer type body
    std::vector<LLType*> types;
    std::vector<LLConstant*>::iterator itr = constants.begin(), end = constants.end();
    for (; itr != end; ++itr)
        types.push_back((*itr)->getType());
    init_type->setBody(types, packed);

    // Whatever type we end up with due to unions, ..., it should match the
    // the LLVM type corresponding to the D struct type in size.
    assert(getTypeStoreSize(DtoType(type)) <= getTypeStoreSize(init_type) &&
        "Struct initializer type mismatch, encountered type too small.");

    // build constant struct
    constInit = LLConstantStruct::get(init_type, constants);
    IF_LOG Logger::cout() << "final default initializer: " << *constInit << std::endl;

    return constInit;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// helper function that returns the static default initializer of a variable
LLConstant* get_default_initializer(VarDeclaration* vd, Initializer* init)
{
    if (init)
    {
        return DtoConstInitializer(init->loc, vd->type, init);
    }
    else if (vd->init)
    {
        return DtoConstInitializer(vd->init->loc, vd->type, vd->init);
    }
    else
    {
        return DtoConstExpInit(vd->loc, vd->type, vd->type->defaultInit(vd->loc));
    }
}

// helper function that adds zero bytes to a vector of constants
size_t add_zeros(std::vector<llvm::Constant*>& constants, size_t diff)
{
    size_t n = constants.size();
    while (diff)
    {
        if (global.params.is64bit && diff % 8 == 0)
        {
            constants.push_back(LLConstant::getNullValue(llvm::Type::getInt64Ty(gIR->context())));
            diff -= 8;
        }
        else if (diff % 4 == 0)
        {
            constants.push_back(LLConstant::getNullValue(llvm::Type::getInt32Ty(gIR->context())));
            diff -= 4;
        }
        else if (diff % 2 == 0)
        {
            constants.push_back(LLConstant::getNullValue(llvm::Type::getInt16Ty(gIR->context())));
            diff -= 2;
        }
        else
        {
            constants.push_back(LLConstant::getNullValue(llvm::Type::getInt8Ty(gIR->context())));
            diff -= 1;
        }
    }
    return constants.size() - n;
}

// Matches the way the type is built in IrTypeStruct
// maybe look at unifying the interface.

std::vector<llvm::Constant*> IrStruct::createStructDefaultInitializer()
{
    IF_LOG Logger::println("Building default initializer for %s", aggrdecl->toPrettyChars());
    LOG_SCOPE;

    assert(type->ty == Tstruct && "cannot build struct default initializer for non struct type");

    DtoType(type);
    IrTypeStruct* ts = stripModifiers(type)->irtype->isStruct();
    assert(ts);

    // start at offset zero
    size_t offset = 0;

    // vector of constants
    std::vector<llvm::Constant*> constants;

    // go through fields
    IrTypeAggr::iterator it;
    for (it = ts->def_begin(); it != ts->def_end(); ++it)
    {
        VarDeclaration* vd = *it;

        assert(vd->offset >= offset && "default fields not sorted by offset");

        IF_LOG Logger::println("using field: %s %s (+%u)", vd->type->toChars(), vd->toChars(), vd->offset);

        // get next aligned offset for this field
        size_t alignedoffset = offset;
        if (!packed)
        {
            alignedoffset = realignOffset(alignedoffset, vd->type);
        }

        // insert explicit padding?
        if (alignedoffset < vd->offset)
        {
            add_zeros(constants, vd->offset - alignedoffset);
        }

        // add initializer
        constants.push_back(get_default_initializer(vd, NULL));

        // advance offset to right past this field
        offset = vd->offset + vd->type->size();
    }

    // tail padding?
    if (offset < aggrdecl->structsize)
    {
        add_zeros(constants, aggrdecl->structsize - offset);
    }

    return constants;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

LLConstant * IrStruct::createStructInitializer(StructInitializer * si)
{
    IF_LOG Logger::println("Building StructInitializer of type %s", si->ad->toPrettyChars());
    LOG_SCOPE;

    // sanity check
    assert(si->ad == aggrdecl && "struct type mismatch");
    assert(si->vars.dim == si->value.dim && "inconsistent StructInitializer");

    // array of things to build
    llvm::SmallVector<IrTypeStruct::VarInitConst, 16> data(aggrdecl->fields.dim);

    // start by creating a map from initializer indices to field indices.
    // I have no fucking clue why dmd represents it like this :/
    size_t n = si->vars.dim;
    LLSmallVector<int, 16> datamap(n, 0);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; 1; j++)
        {
            assert(j < aggrdecl->fields.dim);
            if (aggrdecl->fields.data[j] == si->vars.data[i])
            {
                datamap[i] = j;
                break;
            }
        }
    }

    // fill in explicit initializers
    n = si->vars.dim;
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = static_cast<VarDeclaration*>(si->vars.data[i]);
        Initializer* ini = static_cast<Initializer*>(si->value.data[i]);
        Loc loc = ini ? ini->loc : si->loc;

        size_t idx = datamap[i];

        // check for duplicate initialization
        if (data[idx].first != NULL)
        {
            Loc l = ini ? ini->loc : si->loc;
            error(l, "duplicate initialization of %s", vd->toChars());
            continue;
        }

        // check for overlapping initialization
        for (size_t j = 0; j < i; j++)
        {
            size_t idx2 = datamap[j];
            assert(data[idx2].first);

            VarDeclarationIter it(aggrdecl->fields, idx2);

            unsigned f_begin = it->offset;
            unsigned f_end = f_begin + it->type->size();

            if (vd->offset >= f_end || (vd->offset + vd->type->size()) <= f_begin)
                continue;

            error(loc, "initializer for %s overlaps previous initialization of %s", vd->toChars(), it->toChars());
        }

        IF_LOG Logger::println("Explicit initializer: %s @+%u", vd->toChars(), vd->offset);
        LOG_SCOPE;

        data[idx].first = vd;
        data[idx].second = get_default_initializer(vd, ini);
    }

    // fill in implicit initializers
    n = data.size();
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = data[i].first;
        if (vd)
            continue;

        vd = static_cast<VarDeclaration*>(aggrdecl->fields.data[i]);

        unsigned vd_begin = vd->offset;
        unsigned vd_end = vd_begin + vd->type->size();

        // make sure it doesn't overlap any explicit initializers.
        VarDeclarationIter it(aggrdecl->fields);
        bool overlaps = false;
        size_t j = 0;
        for (; it.more(); it.next(), j++)
        {
            if (i == j || !data[j].first)
                continue;

            unsigned f_begin = it->offset;
            unsigned f_end = f_begin + it->type->size();

            if (vd_begin >= f_end || vd_end <= f_begin)
                continue;

            overlaps = true;
            break;
        }
        // add if no overlap found
        if (!overlaps)
        {
            IF_LOG Logger::println("Implicit initializer: %s @+%u", vd->toChars(), vd->offset);
            LOG_SCOPE;

            data[i].first = vd;
            data[i].second = get_default_initializer(vd, NULL);
        }
    }

    // stop if there were errors
    if (global.errors)
    {
        fatal();
    }

    llvm::Constant* init =
        type->irtype->isStruct()->createInitializerConstant(data, si->ltype);
    si->ltype = static_cast<llvm::StructType*>(init->getType());
    return init;
}
