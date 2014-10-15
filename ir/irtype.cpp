//===-- irtype.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 303
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#else
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#endif
#include "mars.h"
#include "mtype.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irtype.h"

// This code uses llvm::getGlobalContext() as these functions are invoked before gIR is set.
// ... thus it segfaults on gIR==NULL

//////////////////////////////////////////////////////////////////////////////

IrType::IrType(Type* dt, LLType* lt)
:   dtype(dt),
    type(lt)
{
    assert(dt && "null D Type");
    assert(lt && "null LLVM Type");
    assert(!dt->ctype && "already has IrType");
}

//////////////////////////////////////////////////////////////////////////////

IrFuncTy &IrType::getIrFuncTy()
{
    llvm_unreachable("cannot get IrFuncTy from non lazy/function/delegate");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeBasic::IrTypeBasic(Type * dt)
: IrType(dt, basic2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

IrTypeBasic* IrTypeBasic::get(Type* dt)
{
    IrTypeBasic* t = new IrTypeBasic(dt);
    dt->ctype = t;
    return t;
}

//////////////////////////////////////////////////////////////////////////////

LLType* IrTypeBasic::getComplexType(llvm::LLVMContext& ctx, LLType* type)
{
    llvm::Type *types[] = { type, type };
    return llvm::StructType::get(ctx, types, false);
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeBasic::basic2llvm(Type* t)
{
    LLType* t2;

    llvm::LLVMContext& ctx = llvm::getGlobalContext();
    llvm::Triple::ArchType const a = global.params.targetTriple.getArch();
    bool const anyX86 = (a == llvm::Triple::x86) || (a == llvm::Triple::x86_64);
    // Support for doubledouble has been dropped
    //bool const anyPPC = (a == llvm::Triple::ppc) || (a == llvm::Triple::ppc64);

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
        if (anyX86)
            return llvm::Type::getX86_FP80Ty(ctx);
        // Support for doubledouble has been dropped
        // PPC has a special 128bit float
        //else if (anyPPC)
        //    return llvm::Type::getPPC_FP128Ty(ctx);
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
        t2 = anyX86 ? llvm::Type::getX86_FP80Ty(ctx)
            : (anyPPC ? llvm::Type::getPPC_FP128Ty(ctx) : llvm::Type::getDoubleTy(ctx));
        return getComplexType(ctx, t2);

    case Tbool:
        return llvm::Type::getInt1Ty(ctx);
    default:
        llvm_unreachable("Unknown basic type.");
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypePointer::IrTypePointer(Type* dt, LLType* lt)
: IrType(dt, lt)
{
}

//////////////////////////////////////////////////////////////////////////////

IrTypePointer* IrTypePointer::get(Type* dt)
{
    assert(!dt->ctype);
    assert((dt->ty == Tpointer || dt->ty == Tnull) && "not pointer/null type");

    LLType* elemType;
    if (dt->ty == Tnull)
    {
        elemType = llvm::Type::getInt8Ty(llvm::getGlobalContext());
    }
    else
    {
        elemType = i1ToI8(voidToI8(DtoType(dt->nextOf())));

        // DtoType could have already created the same type, e.g. for
        // dt == Node* in struct Node { Node* n; }.
        if (dt->ctype)
            return dt->ctype->isPointer();
    }

    IrTypePointer* t = new IrTypePointer(dt, llvm::PointerType::get(elemType, 0));
    dt->ctype = t;
    return t;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeSArray::IrTypeSArray(Type * dt)
: IrType(dt, sarray2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

IrTypeSArray* IrTypeSArray::get(Type* dt)
{
    IrTypeSArray* t = new IrTypeSArray(dt);
    dt->ctype = t;
    return t;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type * IrTypeSArray::sarray2llvm(Type * t)
{
    assert(t->ty == Tsarray && "not static array type");
    TypeSArray* tsa = static_cast<TypeSArray*>(t);
    uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    LLType* elemType = i1ToI8(voidToI8(DtoType(t->nextOf())));
    return llvm::ArrayType::get(elemType, dim);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeArray::IrTypeArray(Type* dt, LLType* lt)
: IrType(dt, lt)
{
}

//////////////////////////////////////////////////////////////////////////////

IrTypeArray* IrTypeArray::get(Type* dt)
{
    assert(!dt->ctype);
    assert(dt->ty == Tarray && "not dynamic array type");

    LLType* elemType = i1ToI8(voidToI8(DtoType(dt->nextOf())));

    // Could have already built the type as part of a struct forward reference,
    // just as for pointers.
    if (!dt->ctype)
    {
        llvm::Type *types[] = { DtoSize_t(), llvm::PointerType::get(elemType, 0) };
        LLType* at = llvm::StructType::get(llvm::getGlobalContext(), types, false);
        dt->ctype = new IrTypeArray(dt, at);
    }

    return dt->ctype->isArray();
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrTypeVector::IrTypeVector(Type* dt)
: IrType(dt, vector2llvm(dt))
{
}

//////////////////////////////////////////////////////////////////////////////

IrTypeVector* IrTypeVector::get(Type* dt)
{
    IrTypeVector* t = new IrTypeVector(dt);
    dt->ctype = t;
    return t;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Type* IrTypeVector::vector2llvm(Type* dt)
{
    assert(dt->ty == Tvector && "not vector type");
    TypeVector* tv = static_cast<TypeVector*>(dt);
    assert(tv->basetype->ty == Tsarray);
    TypeSArray* tsa = static_cast<TypeSArray*>(tv->basetype);
    uint64_t dim = static_cast<uint64_t>(tsa->dim->toUInteger());
    LLType* elemType = voidToI8(DtoType(tsa->next));
    return llvm::VectorType::get(elemType, dim);
}

//////////////////////////////////////////////////////////////////////////////
