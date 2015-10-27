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
#include "ir/iraggr.h"
#include "irdsymbol.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////

IrAggr::IrAggr(AggregateDeclaration* aggr)
:   aggrdecl(aggr),
    type(aggr->type),
    // above still need to be looked at
    init(0),
    constInit(0),
    init_type(LLStructType::create(gIR->context(), std::string(aggr->toPrettyChars()) + "_init")),
    vtbl(0),
    constVtbl(0),
    classInfo(0),
    constClassInfo(0),
    interfaceVtblMap(),
    classInterfacesArray(0),
    interfacesWithVtbls()
{
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable * IrAggr::getInitSymbol()
{
    if (init)
        return init;

    // create the initZ symbol
    std::string initname("_D");
    initname.append(mangle(aggrdecl));
    initname.append("6__initZ");

    init = getOrCreateGlobal(aggrdecl->loc,
        gIR->module, init_type, true, llvm::GlobalValue::ExternalLinkage, NULL, initname);

    // set alignment
    init->setAlignment(DtoAlignment(type));

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
// FIXME A similar function is in ir/irtypeaggr.cpp
static inline
size_t add_zeros(llvm::SmallVectorImpl<llvm::Constant*>& constants,
    size_t startOffset, size_t endOffset)
{
    assert(startOffset <= endOffset);
    const size_t paddingSize = endOffset - startOffset;
    if (paddingSize)
    {
        llvm::ArrayType* pad = llvm::ArrayType::get(llvm::Type::getInt8Ty(gIR->context()), paddingSize);
        constants.push_back(llvm::Constant::getNullValue(pad));
    }
    return paddingSize ? 1 : 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
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

    if (vd->init)
    {
        return DtoConstInitializer(vd->init->loc, vd->type, vd->init);
    }

    if (vd->type->size(vd->loc) == 0)
    {
        // We need to be able to handle void[0] struct members even if void has
        // no default initializer.
        return llvm::ConstantPointerNull::get(DtoPtrToType(vd->type));
    }
    return DtoConstExpInit(vd->loc, vd->type, vd->type->defaultInit(vd->loc));
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
        offset += Target::ptrsize;

        // add monitor (except for C++ classes)
        if (!aggrdecl->isClassDeclaration()->isCPPclass())
        {
            constants.push_back(getNullValue(getVoidPtrType()));
            offset += Target::ptrsize;
        }
    }

    // Add the initializers for the member fields. While we are traversing the
    // class hierarchy, use the opportunity to populate interfacesWithVtbls if
    // we haven't done so previously (due to e.g. ClassReferenceExp, we can
    // have multiple initializer constants for a single class).
    addFieldInitializers(constants, explicitInitializers, aggrdecl, offset,
        interfacesWithVtbls.empty());

    // tail padding?
    const size_t structsize = aggrdecl->size(Loc());
    if (offset < structsize)
    {
        add_zeros(constants, offset, structsize);
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
            initializerType = LLStructType::get(gIR->context(), types, isPacked());
        else
            initializerType->setBody(types, isPacked());
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
    unsigned& offset,
    bool populateInterfacesWithVtbls
    )
{
    if (ClassDeclaration* cd = decl->isClassDeclaration())
    {
        if (cd->baseClass)
        {
            addFieldInitializers(constants, explicitInitializers,
                cd->baseClass, offset, populateInterfacesWithVtbls);
        }
    }

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

        /* Skip void initializers for unions. DMD bug 3991:
            union X
            {
                int   a = void;
                dchar b = 'a';
            }
        */
        if (decl->isUnionDeclaration() && vd->init && vd->init->isVoidInitializer())
            continue;

        unsigned vd_begin = vd->offset;
        unsigned vd_end = vd_begin + vd->type->size();

        /* Skip zero size fields like zero-length static arrays, LDC issue 812:
            class B {
                ubyte[0] test;
            }
        */
        if (vd_begin == vd_end)
            continue;

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

        // Explicitly zero the padding as per TDPL §7.1.1. Otherwise, it would
        // be left uninitialized by LLVM.
        if (offset < vd->offset)
        {
            add_zeros(constants, offset, vd->offset);
            offset = vd->offset;
        }

        IF_LOG Logger::println("adding field %s", vd->toChars());

        constants.push_back(FillSArrayDims(vd->type, data[i].second));
        offset += getMemberSize(vd->type);
    }

    if (ClassDeclaration* cd = decl->isClassDeclaration())
    {
        // has interface vtbls?
        if (cd->vtblInterfaces && cd->vtblInterfaces->dim > 0)
        {
            // Align interface infos to pointer size.
            unsigned aligned = (offset + Target::ptrsize - 1) & ~(Target::ptrsize - 1);
            if (offset < aligned)
            {
                add_zeros(constants, offset, aligned);
                offset = aligned;
            }

            // false when it's not okay to use functions from super classes
            bool newinsts = (cd == aggrdecl->isClassDeclaration());

            size_t inter_idx = interfacesWithVtbls.size();

            offset = (offset + Target::ptrsize - 1) & ~(Target::ptrsize - 1);

            for (BaseClasses::iterator I = cd->vtblInterfaces->begin(),
                                       E = cd->vtblInterfaces->end();
                                       I != E; ++I)
            {
                constants.push_back(getInterfaceVtbl(*I, newinsts, inter_idx));
                offset += Target::ptrsize;
                inter_idx++;

                if (populateInterfacesWithVtbls)
                    interfacesWithVtbls.push_back(*I);
            }
        }
    }
}

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create)
{
    if (!isIrAggrCreated(decl) && create)
    {
        assert(decl->ir.irAggr == NULL);
        decl->ir.irAggr = new IrAggr(decl);
        decl->ir.m_type = IrDsymbol::AggrType;
    }
    assert(decl->ir.irAggr != NULL);
    return decl->ir.irAggr;
}

bool isIrAggrCreated(AggregateDeclaration *decl)
{
    int t = decl->ir.type();
    assert(t == IrDsymbol::AggrType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::AggrType;
}
