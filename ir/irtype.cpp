#include "llvm/DerivedTypes.h"
#include "ir/irtype.h"

#include "mars.h"
#include "mtype.h"

IrType::IrType(Type* dt, const llvm::Type* lt)
:   dtype(dt),
    pa(lt)
{
    assert(dt && "null D Type");
    assert(lt && "null LLVM Type");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeBasic::IrTypeBasic(Type * dt)
: IrType(dt, basic2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeBasic::basic2llvm(Type* t)
{
    const llvm::Type* t2;

    switch(t->ty)
    {
    case Tvoid:
        return llvm::Type::VoidTy;

    case Tint8:
    case Tuns8:
    case Tchar:
        return llvm::Type::Int8Ty;

    case Tint16:
    case Tuns16:
    case Twchar:
        return llvm::Type::Int16Ty;

    case Tint32:
    case Tuns32:
    case Tdchar:
        return llvm::Type::Int32Ty;

    case Tint64:
    case Tuns64:
        return llvm::Type::Int64Ty;

    /*
    case Tint128:
    case Tuns128:
        return llvm::IntegerType::get(128);
    */

    case Tfloat32:
    case Timaginary32:
        return llvm::Type::FloatTy;

    case Tfloat64:
    case Timaginary64:
        return llvm::Type::DoubleTy;

    case Tfloat80:
    case Timaginary80:
        // only x86 has 80bit float
        if (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            return llvm::Type::X86_FP80Ty;
        // other platforms use 64bit reals
        else
            return llvm::Type::DoubleTy;

    case Tcomplex32:
        t2 = llvm::Type::FloatTy;
        return llvm::StructType::get(t2, t2, NULL);

    case Tcomplex64:
        t2 = llvm::Type::DoubleTy;
        return llvm::StructType::get(t2, t2, NULL);

    case Tcomplex80:
        t2 = (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            ? llvm::Type::X86_FP80Ty
            : llvm::Type::DoubleTy;
        return llvm::StructType::get(t2, t2, NULL);

    case Tbool:
        return llvm::Type::Int1Ty;
    }

    assert(0 && "not basic type");
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypePointer::IrTypePointer(Type * dt)
: IrType(dt, pointer2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

extern const llvm::Type* DtoType(Type* dt);

const llvm::Type * IrTypePointer::pointer2llvm(Type * t)
{
    assert(t->ty == Tpointer && "not pointer type");

    const llvm::Type* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::VoidTy)
        elemType = llvm::Type::Int8Ty;
    return llvm::PointerType::get(elemType, 0);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type * dt)
: IrType(dt, sarray2llvm(dt))
{
    TypeSArray* tsa = (TypeSArray*)dt;
    uint64_t d = (uint64_t)tsa->dim->toUInteger();
    assert(d == dim);
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeSArray::sarray2llvm(Type * t)
{
    assert(t->ty == Tsarray && "not static array type");

    TypeSArray* tsa = (TypeSArray*)t;
    dim = (uint64_t)tsa->dim->toUInteger();
    const llvm::Type* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::VoidTy)
        elemType = llvm::Type::Int8Ty;
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

extern const llvm::Type* DtoSize_t();

const llvm::Type * IrTypeArray::array2llvm(Type * t)
{
    assert(t->ty == Tarray && "not dynamic array type");

    const llvm::Type* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::VoidTy)
        elemType = llvm::Type::Int8Ty;
    elemType = llvm::PointerType::get(elemType, 0);
    return llvm::StructType::get(DtoSize_t(), elemType, NULL);
}

//////////////////////////////////////////////////////////////////////////////

