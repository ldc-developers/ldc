//===-- iraggr.cpp --------------------------------------------------------===//
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
#include "target.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "ir/iraggr.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////

IrAggr::IrAggr(AggregateDeclaration* aggr)
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

LLGlobalVariable * IrAggr::getInitSymbol()
{
    if (init)
        return init;

    // create the initZ symbol
    std::string initname("_D");
    initname.append(aggrdecl->mangle());
    initname.append("6__initZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(aggrdecl);

    init = getOrCreateGlobal(aggrdecl->loc,
        *gIR->module, init_type, true, _linkage, NULL, initname);

    // set alignment
    init->setAlignment(type->alignsize());
    StructDeclaration *sd = aggrdecl->isStructDeclaration();
    if (sd && sd->alignment != STRUCTALIGN_DEFAULT)
        init->setAlignment(sd->alignment);

    return init;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Constant * IrAggr::getDefaultInit()
{
    if (constInit)
        return constInit;

    IF_LOG Logger::println("Building default initializer for %s", aggrdecl->toPrettyChars());
    LOG_SCOPE;

    DtoType(type);
    VarInitMap noExplicitInitializers;
    constInit = createInitializerConstant(noExplicitInitializers, init_type);
    return constInit;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// helper function that adds zero bytes to a vector of constants
size_t add_zeros(llvm::SmallVectorImpl<llvm::Constant*>& constants, size_t diff)
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

LLConstant * IrAggr::createStructInitializer(StructInitializer * si)
{
    IF_LOG Logger::println("Building StructInitializer of type %s", si->ad->toPrettyChars());
    LOG_SCOPE;

    // sanity check
    assert(si->ad == aggrdecl && "struct type mismatch");
    assert(si->vars.dim == si->value.dim && "inconsistent StructInitializer");

    // array of things to build
    VarInitMap initConsts;

    // fill in explicit initializers
    const size_t n = si->vars.dim;
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = si->vars[i];
        Initializer* ini = si->value[i];
        if (!ini)
        {
            // Unclear when this occurs - createInitializerConstant will just
            // fill in default initializer.
            continue;
        }

        VarInitMap::iterator it, end = initConsts.end();
        for (it = initConsts.begin(); it != end; ++it)
        {
            if (it->first == vd)
            {
                error(ini->loc, "duplicate initialization of %s", vd->toChars());
                continue;
            }

            const unsigned f_begin = it->first->offset;
            const unsigned f_end = f_begin + it->first->type->size();

            if (vd->offset < f_end && (vd->offset + vd->type->size()) > f_begin)
            {
                error(ini->loc, "initializer for %s overlaps previous initialization of %s",
                      vd->toChars(), it->first->toChars());
            }
        }

        IF_LOG Logger::println("Explicit initializer: %s @+%u", vd->toChars(), vd->offset);
        LOG_SCOPE;

        initConsts[vd] = DtoConstInitializer(ini->loc, vd->type, ini);
    }
    // stop if there were errors
    if (global.errors)
    {
        fatal();
    }

    llvm::Constant* init = createInitializerConstant(initConsts, si->ltype);
    si->ltype = static_cast<llvm::StructType*>(init->getType());
    return init;
}

//////////////////////////////////////////////////////////////////////////////

typedef std::pair<VarDeclaration*, llvm::Constant*> VarInitConst;

static bool struct_init_data_sort(const VarInitConst& a, const VarInitConst& b)
{
    return (a.first && b.first)
        ? a.first->offset < b.first->offset
        : false;
}

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

// return a constant array of type arrTypeD initialized with a constant value, or that constant value
static llvm::Constant* FillSArrayDims(Type* arrTypeD, llvm::Constant* init)
{
    // Check whether we actually need to expand anything.
    // KLUDGE: We don't have the initializer type here, so we can only check
    // the size without doing an expensive recursive D <-> LLVM type comparison.
    // The better way to solve this would be to just fix the initializer
    // codegen in any place where a scalar initializer might still be generated.
    if (gDataLayout->getTypeStoreSize(init->getType()) >= arrTypeD->size())
        return init;

    if (arrTypeD->ty == Tsarray)
    {
        init = FillSArrayDims(arrTypeD->nextOf(), init);
        size_t dim = static_cast<TypeSArray*>(arrTypeD)->dim->toUInteger();
        llvm::ArrayType* arrty = llvm::ArrayType::get(init->getType(), dim);
        return llvm::ConstantArray::get(arrty,
            std::vector<llvm::Constant*>(dim, init));
    }
    return init;
}

llvm::Constant* IrAggr::createInitializerConstant(
    const VarInitMap& explicitInitializers,
    llvm::StructType* initializerType)
{
    IF_LOG Logger::println("Creating initializer constant for %s", aggrdecl->toChars());
    LOG_SCOPE;

    llvm::SmallVector<llvm::Constant*, 16> constants;

    unsigned offset = 0;
    if (type->ty == Tclass)
    {
        // add vtbl
        constants.push_back(getVtblSymbol());
        // add monitor
        constants.push_back(getNullValue(DtoType(Type::tvoid->pointerTo())));

        // we start right after the vtbl and monitor
        offset = Target::ptrsize * 2;
    }

    addFieldInitializers(constants, explicitInitializers, aggrdecl, offset);

    // tail padding?
    const size_t structsize = type->size();
    if (offset < structsize)
    {
        size_t diff = structsize - offset;
        IF_LOG Logger::println("adding %zu bytes zero padding", diff);
        add_zeros(constants, diff);
    }

    // get initializer type
    if (!initializerType || initializerType->isOpaque())
    {
        llvm::SmallVector<llvm::Constant*, 16>::iterator itr, end = constants.end();
        llvm::SmallVector<llvm::Type*, 16> types;
        types.reserve(constants.size());
        for (itr = constants.begin(); itr != end; ++itr)
            types.push_back((*itr)->getType());
        if (!initializerType)
            initializerType = LLStructType::get(gIR->context(), types, packed);
        else
            initializerType->setBody(types, packed);
    }

    // build constant
    assert(!constants.empty());
    llvm::Constant* c = LLConstantStruct::get(initializerType, constants);
    IF_LOG Logger::cout() << "final initializer: " << *c << std::endl;
    return c;
}

void IrAggr::addFieldInitializers(
    llvm::SmallVectorImpl<llvm::Constant*>& constants,
    const VarInitMap& explicitInitializers,
    AggregateDeclaration* decl,
    unsigned& offset)
{
    if (ClassDeclaration* cd = decl->isClassDeclaration())
    {
        if (cd->baseClass)
        {
            addFieldInitializers(constants, explicitInitializers,
                cd->baseClass, offset);
        }
    }

    const bool packed = (type->ty == Tstruct)
        ? type->alignsize() == 1
        : false;

    // Build up vector with one-to-one mapping to field indices.
    const size_t n = decl->fields.dim;
    llvm::SmallVector<VarInitConst, 16> data(n);

    // Fill in explicit initializers.
    for (size_t i = 0; i < n; ++i)
    {
        VarDeclaration* vd = decl->fields[i];
        VarInitMap::const_iterator expl = explicitInitializers.find(vd);
        if (expl != explicitInitializers.end())
            data[i] = *expl;
    }

    // Fill in implicit initializers
    for (size_t i = 0; i < n; i++)
    {
        if (data[i].first) continue;

        VarDeclaration* vd = decl->fields[i];

        unsigned vd_begin = vd->offset;
        unsigned vd_end = vd_begin + vd->type->size();

        // make sure it doesn't overlap any explicit initializers.
        bool overlaps = false;
        if (type->ty == Tstruct)
        {
            // Only structs and unions can have overlapping fields.
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j || !data[j].first)
                    continue;

                VarDeclaration* it = decl->fields[j];
                unsigned f_begin = it->offset;
                unsigned f_end = f_begin + it->type->size();

                if (vd_begin >= f_end || vd_end <= f_begin)
                    continue;

                overlaps = true;
                break;
            }
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

    // Sort data array by offset.
    // TODO: Figure out whether this is really necessary, fields should already
    // be in offset order. Not having do do this would mean we could use a plain
    // llvm::Constant* vector for initializers and avoid all the VarInitConst business.
    std::sort(data.begin(), data.end(), struct_init_data_sort);

    // build array of constants and make sure explicit zero padding is inserted when necessary.
    for (size_t i = 0; i < n; i++)
    {
        VarDeclaration* vd = data[i].first;
        if (vd == NULL)
            continue;

        // get next aligned offset for this field
        size_t alignedoffset = offset;
        if (!packed)
        {
            alignedoffset = realignOffset(alignedoffset, vd->type);
        }

        // insert explicit padding?
        if (alignedoffset < vd->offset)
        {
            size_t diff = vd->offset - alignedoffset;
            IF_LOG Logger::println("adding %zu bytes zero padding", diff);
            add_zeros(constants, diff);
        }

        IF_LOG Logger::println("adding field %s", vd->toChars());

        constants.push_back(FillSArrayDims(vd->type, data[i].second));
        offset = vd->offset + vd->type->size();
    }

    if (ClassDeclaration* cd = decl->isClassDeclaration())
    {
        // has interface vtbls?
        if (cd->vtblInterfaces && cd->vtblInterfaces->dim > 0)
        {
            // false when it's not okay to use functions from super classes
            bool newinsts = (cd == aggrdecl->isClassDeclaration());

            size_t inter_idx = interfacesWithVtbls.size();

            offset = (offset + Target::ptrsize - 1) & ~(Target::ptrsize - 1);

            ArrayIter<BaseClass> it2(*cd->vtblInterfaces);
            for (; !it2.done(); it2.next())
            {
                BaseClass* b = it2.get();
                constants.push_back(getInterfaceVtbl(b, newinsts, inter_idx));
                offset += Target::ptrsize;

                // add to the interface list
                interfacesWithVtbls.push_back(b);
                inter_idx++;
            }
        }
    }
}
