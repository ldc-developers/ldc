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

extern const llvm::Type* DtoType(Type* dt);
extern const llvm::Type* DtoSize_t();

//////////////////////////////////////////////////////////////////////////////

IrType::IrType(Type* dt, const llvm::Type* lt)
:   dtype(dt),
    pa(lt)
{
    assert(dt && "null D Type");
    assert(lt && "null LLVM Type");
    assert(dt->irtype == NULL && "already has IrType");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeBasic::IrTypeBasic(Type * dt)
: IrType(dt, basic2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeBasic::buildType()
{
    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeBasic::basic2llvm(Type* t)
{
    const llvm::Type* t2;

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

    case Tcomplex32:
        t2 = llvm::Type::getFloatTy(ctx);
        return llvm::StructType::get(ctx, t2, t2, NULL);

    case Tcomplex64:
        t2 = llvm::Type::getDoubleTy(ctx);
        return llvm::StructType::get(ctx, t2, t2, NULL);

    case Tcomplex80:
        t2 = (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            ? llvm::Type::getX86_FP80Ty(ctx)
            : llvm::Type::getDoubleTy(ctx);
        return llvm::StructType::get(ctx, t2, t2, NULL);

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
: IrType(dt, llvm::OpaqueType::get(llvm::getGlobalContext()))
{
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypePointer::buildType()
{
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(
        pointer2llvm(dtype));
    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypePointer::pointer2llvm(Type * dt)
{
    assert(dt->ty == Tpointer && "not pointer type");

    const llvm::Type* elemType = DtoType(dt->nextOf());
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    return llvm::PointerType::get(elemType, 0);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type * dt)
: IrType(dt, llvm::OpaqueType::get(llvm::getGlobalContext()))
{
    assert(dt->ty == Tsarray && "not static array type");
    TypeSArray* tsa = (TypeSArray*)dt;
    dim = (uint64_t)tsa->dim->toUInteger();
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeSArray::buildType()
{
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(
        sarray2llvm(dtype));
    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeSArray::sarray2llvm(Type * t)
{
    const llvm::Type* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    return llvm::ArrayType::get(elemType, dim);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeArray::IrTypeArray(Type * dt)
: IrType(dt, llvm::OpaqueType::get(llvm::getGlobalContext()))
{
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeArray::buildType()
{
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(
        array2llvm(dtype));
    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////

const llvm::Type * IrTypeArray::array2llvm(Type * t)
{
    assert(t->ty == Tarray && "not dynamic array type");

    // get .ptr type
    const llvm::Type* elemType = DtoType(t->nextOf());
    if (elemType == llvm::Type::getVoidTy(llvm::getGlobalContext()))
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    elemType = llvm::PointerType::get(elemType, 0);

    // create struct type
    const llvm::Type* at = llvm::StructType::get(llvm::getGlobalContext(), DtoSize_t(), elemType, NULL);

    // name dynamic array types
    Type::sir->getState()->module->addTypeName(t->toChars(), at);

    return at;
}

//////////////////////////////////////////////////////////////////////////////

