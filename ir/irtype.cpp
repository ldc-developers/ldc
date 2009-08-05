#include "llvm/DerivedTypes.h"
#include "mars.h"
#include "mtype.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "ir/irtype.h"

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
        return llvm::StructType::get(gIR->context(), t2, t2, NULL);

    case Tcomplex64:
        t2 = llvm::Type::DoubleTy;
        return llvm::StructType::get(gIR->context(), t2, t2, NULL);

    case Tcomplex80:
        t2 = (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            ? llvm::Type::X86_FP80Ty
            : llvm::Type::DoubleTy;
        return llvm::StructType::get(gIR->context(), t2, t2, NULL);

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
: IrType(dt, llvm::OpaqueType::get())
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
    if (elemType == llvm::Type::VoidTy)
        elemType = llvm::Type::Int8Ty;
    return llvm::PointerType::get(elemType, 0);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type * dt)
: IrType(dt, llvm::OpaqueType::get())
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
    if (elemType == llvm::Type::VoidTy)
        elemType = llvm::Type::Int8Ty;
    return llvm::ArrayType::get(elemType, dim);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeArray::IrTypeArray(Type * dt)
: IrType(dt, llvm::OpaqueType::get())
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
    if (elemType == llvm::Type::VoidTy)
        elemType = llvm::Type::Int8Ty;
    elemType = llvm::PointerType::get(elemType, 0);

    // create struct type
    const llvm::Type* at = llvm::StructType::get(gIR->context(), DtoSize_t(), elemType, NULL);

    // name dynamic array types
    Type::sir->getState()->module->addTypeName(t->toChars(), at);

    return at;
}

//////////////////////////////////////////////////////////////////////////////

