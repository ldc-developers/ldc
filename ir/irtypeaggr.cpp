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
#include "declaration.h"
#include "init.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/utils.h"
#include "gen/llvmhelpers.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeAggr::IrTypeAggr(AggregateDeclaration * ad)
:   IrType(ad->type, LLStructType::create(gIR->context(), ad->toPrettyChars())),
    aggr(ad)
{
}

//////////////////////////////////////////////////////////////////////////////

static bool struct_init_data_sort(const IrTypeAggr::VarInitConst& a,
                                  const IrTypeAggr::VarInitConst& b)
{
    return (a.first && b.first)
        ? a.first->offset < b.first->offset
        : false;
}

extern size_t add_zeros(std::vector<llvm::Constant*>& constants, size_t diff);

// return a constant array of type arrTypeD initialized with a constant value, or that constant value
static llvm::Constant* FillSArrayDims(Type* arrTypeD, llvm::Constant* init)
{
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

llvm::Constant* IrTypeAggr::createInitializerConstant(
    llvm::ArrayRef<IrTypeAggr::VarInitConst> initializers,
    llvm::StructType* initializerType)
{
    const bool packed = (dtype->ty == Tstruct)
        ? dtype->alignsize() == 1
        : false;

    const size_t n = initializers.size();

    // sort data array by offset
    llvm::SmallVector<IrTypeAggr::VarInitConst, 16> data(
        initializers.begin(), initializers.end());
    std::sort(data.begin(), data.end(), struct_init_data_sort);

    // build array of constants and make sure explicit zero padding is inserted when necessary.
    size_t offset = 0;
    std::vector<llvm::Constant*> constants;
    constants.reserve(n);

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

    // tail padding?
    const size_t structsize = getTypePaddedSize(type);
    if (offset < structsize)
    {
        size_t diff = structsize - offset;
        IF_LOG Logger::println("adding %zu bytes zero padding", diff);
        add_zeros(constants, diff);
    }

    // get initializer type
    if (!initializerType || initializerType->isOpaque())
    {
        std::vector<llvm::Constant*>::iterator itr, end = constants.end();
        std::vector<llvm::Type*> types;
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
    IF_LOG Logger::cout() << "final struct initializer: " << *c << std::endl;
    return c;
}
