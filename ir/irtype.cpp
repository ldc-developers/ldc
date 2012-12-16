//===-- irtype.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "mars.h"
#include "mtype.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "ir/irtype.h"

// This code uses llvm::getGlobalContext() as these functions are invoked before gIR is set.
// ... thus it segfaults on gIR==NULL

//////////////////////////////////////////////////////////////////////////////

extern LLType* DtoType(Type* dt);
extern LLIntegerType* DtoSize_t();

//////////////////////////////////////////////////////////////////////////////

IrType::IrType(Type* dt, LLType* lt)
:   dtype(dt),
    type(lt)
{
    assert(dt && "null D Type");
    assert(lt && "null LLVM Type");
#if 0
    // FIXME: For some reason the assert fails
    assert(dt->irtype == NULL && "already has IrType");
#endif
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeBasic::IrTypeBasic(Type * dt)
: IrType(dt, basic2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeBasic::buildType()
{
    return type;
}

//////////////////////////////////////////////////////////////////////////////

LLType* IrTypeBasic::getComplexType(llvm::LLVMContext& ctx, LLType* type)
{
    llvm::SmallVector<LLType*, 2> types;
    types.push_back(type);
    types.push_back(type);
    return llvm::StructType::get(ctx, types);
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeBasic::basic2llvm(Type* t)
{
    LLType* t2;

    llvm::LLVMContext& ctx = llvm::getGlobalContext();

    switch(t->ty)
    {
    case Tvoid:
        return llvm::Type::getVoidTy(ctx);

    case Tint8:
    case Tuns8:
    case Tchar:
        return llvm::Type::getInt8Ty(ctx);

    case Tint16:
    case Tuns16:
    case Twchar:
        return llvm::Type::getInt16Ty(ctx);

    case Tint32:
    case Tuns32:
    case Tdchar:
        return llvm::Type::getInt32Ty(ctx);

    case Tint64:
    case Tuns64:
        return llvm::Type::getInt64Ty(ctx);

    /*
    case Tint128:
    case Tuns128:
        return llvm::IntegerType::get(llvm::getGlobalContext(), 128);
    */

    case Tfloat32:
    case Timaginary32:
        return llvm::Type::getFloatTy(ctx);

    case Tfloat64:
    case Timaginary64:
        return llvm::Type::getDoubleTy(ctx);

    case Tfloat80:
    case Timaginary80:
        // only x86 has 80bit float
        if (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            return llvm::Type::getX86_FP80Ty(ctx);
        // other platforms use 64bit reals
        else
            return llvm::Type::getDoubleTy(ctx);

    case Tcomplex32: {
        t2 = llvm::Type::getFloatTy(ctx);
        return getComplexType(ctx, t2);
    }

    case Tcomplex64:
        t2 = llvm::Type::getDoubleTy(ctx);
        return getComplexType(ctx, t2);

    case Tcomplex80:
        t2 = (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            ? llvm::Type::getX86_FP80Ty(ctx)
            : llvm::Type::getDoubleTy(ctx);
        return getComplexType(ctx, t2);

    case Tbool:
        return llvm::Type::getInt1Ty(ctx);
    default:
        assert(0 && "not basic type");
        return NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypePointer::IrTypePointer(Type * dt)
: IrType(dt, dt->ty == Tnull ? null2llvm(dt) : pointer2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypePointer::buildType()
{
    return type;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypePointer::pointer2llvm(Type * dt)
{
    assert(dt->ty == Tpointer && "not pointer type");

    LLType* elemType = DtoType(dt->nextOf());
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    return llvm::PointerType::get(elemType, 0);
}

llvm::Type* IrTypePointer::null2llvm(Type* dt)
{
    assert(dt->ty == Tnull && "not null type");
    LLType* elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    return llvm::PointerType::get(elemType, 0);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type * dt)
: IrType(dt, sarray2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeSArray::buildType()
{
    return type;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeSArray::sarray2llvm(Type * t)
{
    assert(t->ty == Tsarray && "not static array type");
    TypeSArray* tsa = static_cast<TypeSArray*>(t);
    dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    LLType* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    return llvm::ArrayType::get(elemType, dim);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeArray::IrTypeArray(Type * dt)
: IrType(dt, array2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeArray::buildType()
{
    return type;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeArray::array2llvm(Type * t)
{
    assert(t->ty == Tarray && "not dynamic array type");

    // get .ptr type
    LLType* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    elemType = llvm::PointerType::get(elemType, 0);

    // create struct type
    llvm::SmallVector<LLType*, 2> types;
    types.push_back(DtoSize_t());
    types.push_back(elemType);
    LLType* at = llvm::StructType::get(llvm::getGlobalContext(), types/*, t->toChars()*/);

    return at;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#if DMDV2

IrTypeVector::IrTypeVector(Type* dt)
: IrType(dt, vector2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type* IrTypeVector::buildType()
{
    return type;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type* IrTypeVector::vector2llvm(Type* dt)
{
    assert(dt->ty == Tvector && "not vector type");
    TypeVector* tv = static_cast<TypeVector*>(dt);
    assert(tv->basetype->ty == Tsarray);
    TypeSArray* tsa = static_cast<TypeSArray*>(tv->basetype);
    dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    LLType* elemType = DtoType(tsa->next);
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    return llvm::VectorType::get(elemType, dim);
}

#endif

//////////////////////////////////////////////////////////////////////////////
