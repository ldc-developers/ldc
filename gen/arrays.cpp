//===-- arrays.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/arrays.h"
#include "aggregate.h"
#include "declaration.h"
#include "dsymbol.h"
#include "expression.h"
#include "init.h"
#include "module.h"
#include "mtype.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irmodule.h"

//////////////////////////////////////////////////////////////////////////////////////////

static LLValue *DtoSlice(DValue *dval)
{
    LLValue *val = dval->getRVal();
    if (dval->getType()->toBasetype()->ty == Tsarray) {
        // Convert static array to slice
        LLStructType *type = DtoArrayType(LLType::getInt8Ty(gIR->context()));
        LLValue *array = DtoRawAlloca(type, 0, ".array");
        DtoStore(DtoArrayLen(dval), DtoGEPi(array, 0, 0, ".len"));
        DtoStore(DtoBitCast(val, getVoidPtrType()), DtoGEPi(array, 0, 1, ".ptr"));
        val = DtoLoad(array);
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

static LLValue *DtoSlicePtr(DValue *dval)
{
    Loc loc;
    LLStructType *type = DtoArrayType(LLType::getInt8Ty(gIR->context()));
    Type *vt = dval->getType()->toBasetype();
    if (vt->ty == Tarray)
        return makeLValue(loc, dval);

    bool isStaticArray = vt->ty == Tsarray;
    LLValue *val = isStaticArray ? dval->getRVal() : makeLValue(loc, dval);
    LLValue *array = DtoRawAlloca(type, 0, ".array");
    LLValue *len = isStaticArray ? DtoArrayLen(dval) : DtoConstSize_t(1);
    DtoStore(len, DtoGEPi(array, 0, 0, ".len"));
    DtoStore(DtoBitCast(val, getVoidPtrType()), DtoGEPi(array, 0, 1, ".ptr"));
    return array;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLStructType* DtoArrayType(Type* arrayTy)
{
    assert(arrayTy->nextOf());
    LLType* elemty = i1ToI8(voidToI8(DtoType(arrayTy->nextOf())));

    llvm::Type *elems[] = { DtoSize_t(), getPtrToType(elemty) };
    return llvm::StructType::get(gIR->context(), elems, false);
}

LLStructType* DtoArrayType(LLType* t)
{
    llvm::Type *elems[] = { DtoSize_t(), getPtrToType(t) };
    return llvm::StructType::get(gIR->context(), elems, false);
}

//////////////////////////////////////////////////////////////////////////////////////////

LLArrayType* DtoStaticArrayType(Type* t)
{
    t = t->toBasetype();
    assert(t->ty == Tsarray);
    TypeSArray* tsa = static_cast<TypeSArray*>(t);
    Type* tnext = tsa->nextOf();

    LLType* elemty = i1ToI8(voidToI8(DtoType(tnext)));
    return LLArrayType::get(elemty, tsa->dim->toUInteger());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoSetArrayToNull(LLValue* v)
{
    IF_LOG Logger::println("DtoSetArrayToNull");
    LOG_SCOPE;

    assert(isaPointer(v));
    LLType* t = v->getType()->getContainedType(0);

    DtoStore(LLConstant::getNullValue(t), v);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoArrayInit(Loc& loc, DValue* array, DValue* value, int op)
{
    IF_LOG Logger::println("DtoArrayInit");
    LOG_SCOPE;

    if (op != -1 && op != TOKblit && arrayNeedsPostblit(array->type))
    {
        DtoArraySetAssign(loc, array, value, op);
        return;
    }

    LLValue* dim = DtoArrayLen(array);
    LLValue* ptr = DtoArrayPtr(array);
    Type* arrayelemty = array->getType()->nextOf()->toBasetype();

    // lets first optimize all zero/constant i8 initializations down to a memset.
    // this simplifies codegen later on as llvm null's have no address!
    LLValue *val = value->getRVal();
    if (isaConstant(val) && (isaConstant(val)->isNullValue()
                             || val->getType() == LLType::getInt8Ty(gIR->context())))
    {
        size_t X = getTypePaddedSize(val->getType());
        LLValue* nbytes = gIR->ir->CreateMul(dim, DtoConstSize_t(X), ".nbytes");
        if (isaConstant(val)->isNullValue())
            DtoMemSetZero(ptr, nbytes);
        else
            DtoMemSet(ptr, val, nbytes);
        return;
    }

    // create blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* condbb = llvm::BasicBlock::Create(gIR->context(), "arrayinit.cond",
                                                        gIR->topfunc(), oldend);
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(gIR->context(), "arrayinit.body",
                                                        gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb  = llvm::BasicBlock::Create(gIR->context(), "arrayinit.end",
                                                        gIR->topfunc(), oldend);

    // initialize iterator
    LLValue *itr = DtoAlloca(Type::tsize_t, "arrayinit.itr");
    DtoStore(DtoConstSize_t(0), itr);

    // move into the for condition block, ie. start the loop
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(condbb, gIR->scopebb());

    // replace current scope
    gIR->scope() = IRScope(condbb,bodybb);

    // create the condition
    LLValue* cond_val = gIR->ir->CreateICmpNE(DtoLoad(itr), dim, "arrayinit.condition");

    // conditional branch
    assert(!gIR->scopereturned());
    llvm::BranchInst::Create(bodybb, endbb, cond_val, gIR->scopebb());

    // rewrite scope
    gIR->scope() = IRScope(bodybb, endbb);

    // assign array element value
    DValue *arrayelem = new DVarValue(arrayelemty, DtoGEP1(ptr, DtoLoad(itr), "arrayinit.arrayelem"));
    DtoAssign(loc, arrayelem, value, op);

    // increment iterator
    DtoStore(gIR->ir->CreateAdd(DtoLoad(itr), DtoConstSize_t(1), "arrayinit.new_itr"), itr);

    // loop
    llvm::BranchInst::Create(condbb, gIR->scopebb());

    // rewrite the scope
    gIR->scope() = IRScope(endbb, oldend);
}

//////////////////////////////////////////////////////////////////////////////////////////

Type *DtoArrayElementType(Type *arrayType)
{
    assert(arrayType->toBasetype()->nextOf());
    Type *t = arrayType->toBasetype()->nextOf()->toBasetype();
    while (t->ty == Tsarray)
        t = t->nextOf()->toBasetype();
    return t;
}

// Determine whether t is an array of structs that need a postblit.
bool arrayNeedsPostblit(Type *t)
{
    t = DtoArrayElementType(t);
    if (t->ty == Tstruct)
        return static_cast<TypeStruct *>(t)->sym->postblit != 0;
    return false;
}

// Does array assignment (or initialization) from another array of the same element type.
void DtoArrayAssign(Loc& loc, DValue *array, DValue *value, int op)
{
    IF_LOG Logger::println("DtoArrayAssign");
    LOG_SCOPE;

    assert(value && array);
    assert(op != TOKblit);

    // Use array->type instead of value->type so as to not accidentally pick
    // up a superfluous const layer (TypeInfo_Const doesn't pass on postblit()).
    Type *t = array->type->toBasetype();
    assert(t->nextOf());
    Type *elemType = t->nextOf()->toBasetype();

    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, op == TOKconstruct ? "_d_arrayctor" : "_d_arrayassign");
    LLValue* args[] = {
        DtoTypeInfoOf(elemType),
        DtoAggrPaint(DtoSlice(value), fn->getFunctionType()->getParamType(1)),
        DtoAggrPaint(DtoSlice(array), fn->getFunctionType()->getParamType(2))
    };

    LLCallSite call = gIR->CreateCallOrInvoke(fn, args, ".array");
    call.setCallingConv(llvm::CallingConv::C);
}

// If op is TOKconstruct, does construction of an array;
// otherwise, does assignment to an array.
void DtoArraySetAssign(Loc& loc, DValue *array, DValue *value, int op)
{
    IF_LOG Logger::println("DtoArraySetAssign");
    LOG_SCOPE;

    assert(array && value);
    assert(op != TOKblit);

    LLValue* ptr = DtoArrayPtr(array);
    LLValue* len = DtoArrayLen(array);
    // The count parameter is of type int but len is of type size_t.
    // If size_t is not int then truncated the value.
    LLType* intType = LLType::getInt32Ty(gIR->context());
    if (len->getType() != intType)
        len = gIR->ir->CreateTrunc(len, intType);

    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, op == TOKconstruct ? "_d_arraysetctor" : "_d_arraysetassign");
    LLValue* args[] = {
        DtoBitCast(ptr, getVoidPtrType()),
        DtoBitCast(makeLValue(loc, value), getVoidPtrType()),
        len,
        DtoTypeInfoOf(array->type->toBasetype()->nextOf()->toBasetype())
    };

    LLCallSite call = gIR->CreateCallOrInvoke(fn, args, ".newptr");
    call.setCallingConv(llvm::CallingConv::C);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoSetArray(DValue* array, LLValue* dim, LLValue* ptr)
{
    IF_LOG Logger::println("SetArray");
    LLValue *arr = array->getLVal();
    assert(isaStruct(arr->getType()->getContainedType(0)));
    DtoStore(dim, DtoGEPi(arr,0,0));
    DtoStore(ptr, DtoGEPi(arr,0,1));
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstArrayInitializer(ArrayInitializer* arrinit)
{
    IF_LOG Logger::println("DtoConstArrayInitializer: %s | %s", arrinit->toChars(), arrinit->type->toChars());
    LOG_SCOPE;

    assert(arrinit->value.dim == arrinit->index.dim);

    // get base array type
    Type* arrty = arrinit->type->toBasetype();
    size_t arrlen = arrinit->dim;

    // for statis arrays, dmd does not include any trailing default
    // initialized elements in the value/index lists
    if (arrty->ty == Tsarray)
    {
        TypeSArray* tsa = static_cast<TypeSArray*>(arrty);
        arrlen = static_cast<size_t>(tsa->dim->toInteger());
    }

    // make sure the number of initializers is sane
    if (arrinit->index.dim > arrlen || arrinit->dim > arrlen)
    {
        error(arrinit->loc, "too many initializers, %llu, for array[%llu]",
                             static_cast<unsigned long long>(arrinit->index.dim),
                             static_cast<unsigned long long>(arrlen));
        fatal();
    }

    // get elem type
    Type* elemty;
    if (arrty->ty == Tvector)
        elemty = static_cast<TypeVector *>(arrty)->elementType();
    else
        elemty = arrty->nextOf();
    LLType* llelemty = i1ToI8(voidToI8(DtoType(elemty)));

    // true if array elements differ in type, can happen with array of unions
    bool mismatch = false;

    // allocate room for initializers
    std::vector<LLConstant*> initvals(arrlen, NULL);

    // go through each initializer, they're not sorted by index by the frontend
    size_t j = 0;
    for (size_t i = 0; i < arrinit->index.dim; i++)
    {
        // get index
        Expression* idx = static_cast<Expression*>(arrinit->index.data[i]);

        // idx can be null, then it's just the next element
        if (idx)
            j = idx->toInteger();
        assert(j < arrlen);

        // get value
        Initializer* val = static_cast<Initializer*>(arrinit->value.data[i]);
        assert(val);

        // error check from dmd
        if (initvals[j] != NULL)
        {
            error(arrinit->loc, "duplicate initialization for index %llu", static_cast<unsigned long long>(j));
        }

        LLConstant* c = DtoConstInitializer(val->loc, elemty, val);
        assert(c);
        if (c->getType() != llelemty)
            mismatch = true;

        initvals[j] = c;
        j++;
    }

    // die now if there was errors
    if (global.errors)
        fatal();

    // fill out any null entries still left with default values

    // element default initializer
    LLConstant* defelem = DtoConstExpInit(arrinit->loc, elemty, elemty->defaultInit(arrinit->loc));
    bool mismatch2 =  (defelem->getType() != llelemty);

    for (size_t i = 0; i < arrlen; i++)
    {
        if (initvals[i] != NULL)
            continue;

        initvals[i] = defelem;

        if (mismatch2)
            mismatch = true;
    }

    LLConstant* constarr;
    if (mismatch)
        constarr = LLConstantStruct::getAnon(gIR->context(), initvals); // FIXME should this pack?
    else
    {
        if (arrty->ty == Tvector)
            constarr = llvm::ConstantVector::get(initvals);
        else
            constarr = LLConstantArray::get(LLArrayType::get(llelemty, arrlen), initvals);
    }

//     std::cout << "constarr: " << *constarr << std::endl;

    // if the type is a static array, we're done
    if (arrty->ty == Tsarray || arrty->ty == Tvector)
        return constarr;

    // we need to make a global with the data, so we have a pointer to the array
    // Important: don't make the gvar constant, since this const initializer might
    // be used as an initializer for a static T[] - where modifying contents is allowed.
    LLGlobalVariable* gvar = new LLGlobalVariable(*gIR->module, constarr->getType(), false, LLGlobalValue::InternalLinkage, constarr, ".constarray");

    if (arrty->ty == Tpointer)
        // we need to return pointer to the static array.
        return DtoBitCast(gvar, DtoType(arrty));

    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };

#if LDC_LLVM_VER >= 307
    LLConstant* gep = llvm::ConstantExpr::getGetElementPtr(isaPointer(gvar)->getElementType(), gvar, idxs, true);
#else
    LLConstant* gep = llvm::ConstantExpr::getGetElementPtr(gvar, idxs, true);
#endif
    gep = llvm::ConstantExpr::getBitCast(gvar, getPtrToType(llelemty));

    return DtoConstSlice(DtoConstSize_t(arrlen), gep, arrty);
}

//////////////////////////////////////////////////////////////////////////////////////////

bool isConstLiteral(ArrayLiteralExp* ale)
{
    // FIXME: This is overly pessemistic, isConst() always returns 0 e.g. for
    // StructLiteralExps. Thus, we waste optimization potential (GitHub #506).
    for (size_t i = 0; i < ale->elements->dim; ++i)
    {
        // We have to check specifically for '1', as SymOffExp is classified as
        // '2' and the address of a local variable is not an LLVM constant.
        if ((*ale->elements)[i]->isConst() != 1)
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* arrayLiteralToConst(IRState* p, ArrayLiteralExp* ale)
{
    // Build the initializer. We have to take care as due to unions in the
    // element types (with different fields being initialized), we can end up
    // with different types for the initializer values. In this case, we
    // generate a packed struct constant instead of an array constant.
    LLType *elementType = NULL;
    bool differentTypes = false;

    std::vector<LLConstant*> vals;
    vals.reserve(ale->elements->dim);
    for (unsigned i = 0; i < ale->elements->dim; ++i)
    {
        llvm::Constant *val = toConstElem((*ale->elements)[i], p);
        if (!elementType)
            elementType = val->getType();
        else
            differentTypes |= (elementType != val->getType());
        vals.push_back(val);
    }

    if (differentTypes)
        return llvm::ConstantStruct::getAnon(vals, true);

    if (!elementType)
    {
        assert(ale->elements->dim == 0);
        elementType = i1ToI8(voidToI8(DtoType(ale->type->toBasetype()->nextOf())));
        return llvm::ConstantArray::get(LLArrayType::get(elementType, 0), vals);
    }

    llvm::ArrayType *t = llvm::ArrayType::get(elementType, ale->elements->dim);
    return llvm::ConstantArray::get(t, vals);
}

//////////////////////////////////////////////////////////////////////////////////////////

void initializeArrayLiteral(IRState* p, ArrayLiteralExp* ale, LLValue* dstMem)
{
    size_t elemCount = ale->elements->dim;

    // Don't try to write nothing to a zero-element array, we might represent it
    // as a null pointer.
    if (elemCount == 0) return;

    if (isConstLiteral(ale))
    {
        llvm::Constant* constarr = arrayLiteralToConst(p, ale);

        // Emit a global for longer arrays, as an inline constant is always
        // lowered to a series of movs or similar at the asm level. The
        // optimizer can still decide to promote the memcpy intrinsic, so
        // the cutoff merely affects compilation speed.
        if (elemCount <= 4)
        {
            DtoStore(constarr, DtoBitCast(dstMem, getPtrToType(constarr->getType())));
        }
        else
        {
            llvm::GlobalVariable* gvar = new llvm::GlobalVariable(
                *gIR->module,
                constarr->getType(),
                true,
                LLGlobalValue::InternalLinkage,
                constarr,
                ".arrayliteral"
            );
            gvar->setUnnamedAddr(true);
            DtoMemCpy(dstMem, gvar, DtoConstSize_t(getTypePaddedSize(constarr->getType())));
        }
    }
    else
    {
        // Store the elements one by one.
        for (size_t i = 0; i < elemCount; ++i)
        {
            DValue* e = toElem((*ale->elements)[i]);

            LLValue* elemAddr = DtoGEPi(dstMem, 0, i, "", p->scopebb());
            DVarValue* vv = new DVarValue(e->type, elemAddr);
            DtoAssign(ale->loc, vv, e);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
static LLValue* get_slice_ptr(DSliceValue* e, LLValue*& sz)
{
    assert(e->len != 0);
    LLType* t = e->ptr->getType()->getContainedType(0);
    sz = gIR->ir->CreateMul(DtoConstSize_t(getTypePaddedSize(t)), e->len);
    return DtoBitCast(e->ptr, getVoidPtrType());
}

static void copySlice(Loc& loc, LLValue* dstarr, LLValue* sz1, LLValue* srcarr, LLValue* sz2)
{
    if (global.params.useAssert || gIR->emitArrayBoundsChecks())
    {
        LLValue* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_array_slice_copy");
        gIR->CreateCallOrInvoke4(fn, dstarr, sz1, srcarr, sz2);
    }
    else
    {
        // We might have dstarr == srcarr at compile time, but as long as
        // sz1 == 0 at runtime, this would probably still be legal (the C spec
        // is unclear here).
        DtoMemCpy(dstarr, srcarr, sz1);
    }
}

void DtoArrayCopySlices(Loc& loc, DSliceValue* dst, DSliceValue* src)
{
    IF_LOG Logger::println("ArrayCopySlices");

    LLValue *sz1,*sz2;
    LLValue* dstarr = get_slice_ptr(dst,sz1);
    LLValue* srcarr = get_slice_ptr(src,sz2);

    copySlice(loc, dstarr, sz1, srcarr, sz2);
}

void DtoArrayCopyToSlice(Loc& loc, DSliceValue* dst, DValue* src)
{
    IF_LOG Logger::println("ArrayCopyToSlice");

    LLValue* sz1;
    LLValue* dstarr = get_slice_ptr(dst,sz1);

    LLValue* srcarr = DtoBitCast(DtoArrayPtr(src), getVoidPtrType());
    LLType* arrayelemty = voidToI8(DtoType(src->getType()->nextOf()->toBasetype()));
    LLValue* sz2 = gIR->ir->CreateMul(DtoConstSize_t(getTypePaddedSize(arrayelemty)), DtoArrayLen(src));

    copySlice(loc, dstarr, sz1, srcarr, sz2);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoStaticArrayCopy(LLValue* dst, LLValue* src)
{
    IF_LOG Logger::println("StaticArrayCopy");

    size_t n = getTypePaddedSize(dst->getType()->getContainedType(0));
    DtoMemCpy(dst, src, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////
LLConstant* DtoConstSlice(LLConstant* dim, LLConstant* ptr, Type *type)
{
    LLConstant* values[2] = { dim, ptr };
    llvm::ArrayRef<LLConstant*> valuesRef = llvm::makeArrayRef(values, 2);
    LLStructType *lltype = type ?
                isaStruct(DtoType(type)) :
                LLConstantStruct::getTypeForElements(gIR->context(), valuesRef);
    return LLConstantStruct::get(lltype, valuesRef);
}

//////////////////////////////////////////////////////////////////////////////////////////
static bool isInitialized(Type* et) {
    // Strip static array types from element type
    Type* bt = et->toBasetype();
    while (bt->ty == Tsarray) {
        et = bt->nextOf();
        bt = et->toBasetype();
    }
    // Otherwise, it's always initialized.
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////

static DSliceValue *getSlice(Type *arrayType, LLValue *array)
{
    // Get ptr and length of the array
    LLValue* arrayLen = DtoExtractValue(array, 0, ".len");
    LLValue* newptr = DtoExtractValue(array, 1, ".ptr");

    // cast pointer to wanted type
    LLType* dstType = DtoType(arrayType)->getContainedType(1);
    if (newptr->getType() != dstType)
        newptr = DtoBitCast(newptr, dstType, ".gc_mem");

    return new DSliceValue(arrayType, arrayLen, newptr);
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoNewDynArray(Loc& loc, Type* arrayType, DValue* dim, bool defaultInit)
{
    IF_LOG Logger::println("DtoNewDynArray : %s", arrayType->toChars());
    LOG_SCOPE;

    // typeinfo arg
    LLValue* arrayTypeInfo = DtoTypeInfoOf(arrayType);

    // dim arg
    assert(DtoType(dim->getType()) == DtoSize_t());
    LLValue* arrayLen = dim->getRVal();

    // get runtime function
    Type* eltType = arrayType->toBasetype()->nextOf();
    if (defaultInit && !isInitialized(eltType))
        defaultInit = false;
    bool zeroInit = eltType->isZeroInit();

    const char* fnname = defaultInit ?
        (zeroInit ? "_d_newarrayT" : "_d_newarrayiT") :
        "_d_newarrayU";
    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, fnname);

    // call allocator
    LLValue* newArray = gIR->CreateCallOrInvoke2(fn, arrayTypeInfo, arrayLen, ".gc_mem").getInstruction();

    return getSlice(arrayType, newArray);
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoNewMulDimDynArray(Loc& loc, Type* arrayType, DValue** dims, size_t ndims)
{
    IF_LOG Logger::println("DtoNewMulDimDynArray : %s", arrayType->toChars());
    LOG_SCOPE;

    // typeinfo arg
    LLValue* arrayTypeInfo = DtoTypeInfoOf(arrayType);

    // get value type
    Type* vtype = arrayType->toBasetype();
    for (size_t i = 0; i < ndims; ++i)
        vtype = vtype->nextOf();

    // get runtime function
    const char* fnname = vtype->isZeroInit() ? "_d_newarraymTX" : "_d_newarraymiTX";
    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, fnname);

    // Check if constant
    bool allDimsConst = true;
    for (size_t i = 0; i < ndims; ++i)
    {
        if (!isaConstant(dims[i]->getRVal()))
            allDimsConst = false;
    }

    // build dims
    LLValue* array;
    if (allDimsConst)
    {
        // Build constant array for dimensions
        std::vector<LLConstant*> argsdims;
        argsdims.reserve(ndims);
        for (size_t i = 0; i < ndims; ++i)
        {
            argsdims.push_back(isaConstant(dims[i]->getRVal()));
        }

        llvm::Constant* dims = llvm::ConstantArray::get(llvm::ArrayType::get(DtoSize_t(), ndims), argsdims);
        LLGlobalVariable* gvar = new llvm::GlobalVariable(*gIR->module, dims->getType(), true, LLGlobalValue::InternalLinkage, dims, ".dimsarray");
        array = llvm::ConstantExpr::getBitCast(gvar, getPtrToType(dims->getType()));
    }
    else
    {
        // Build static array for dimensions
        LLArrayType* type = LLArrayType::get(DtoSize_t(), ndims);
        array = DtoRawAlloca(type, 0, ".dimarray");
        for (size_t i = 0; i < ndims; ++i)
            DtoStore(dims[i]->getRVal(), DtoGEPi(array, 0, i, ".ndim"));
    }

    LLStructType* dtype = DtoArrayType(DtoSize_t());
    LLValue* darray = DtoRawAlloca(dtype, 0, ".array");
    DtoStore(DtoConstSize_t(ndims), DtoGEPi(darray, 0, 0, ".len"));
    DtoStore(DtoBitCast(array, getPtrToType(DtoSize_t())), DtoGEPi(darray, 0, 1, ".ptr"));

    llvm::Value* args[] = {
        arrayTypeInfo,
        DtoLoad(darray)
    };

    // call allocator
    LLValue* newptr = gIR->CreateCallOrInvoke(fn, args, ".gc_mem").getInstruction();

    IF_LOG Logger::cout() << "final ptr = " << *newptr << '\n';

    return getSlice(arrayType, newptr);
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoResizeDynArray(Loc& loc, Type* arrayType, DValue* array, LLValue* newdim)
{
    IF_LOG Logger::println("DtoResizeDynArray : %s", arrayType->toChars());
    LOG_SCOPE;

    assert(array);
    assert(newdim);
    assert(arrayType);
    assert(arrayType->toBasetype()->ty == Tarray);

    // decide on what runtime function to call based on whether the type is zero initialized
    bool zeroInit = arrayType->toBasetype()->nextOf()->isZeroInit();

    // call runtime
    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, zeroInit ? "_d_arraysetlengthT" : "_d_arraysetlengthiT" );

    LLValue* args[] = {
        DtoTypeInfoOf(arrayType),
        newdim,
        DtoBitCast(array->getLVal(), fn->getFunctionType()->getParamType(2))
    };
    LLValue* newArray = gIR->CreateCallOrInvoke(fn, args, ".gc_mem").getInstruction();

    return getSlice(arrayType, newArray);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoCatAssignElement(Loc& loc, Type* arrayType, DValue* array, Expression* exp)
{
    IF_LOG Logger::println("DtoCatAssignElement");
    LOG_SCOPE;

    assert(array);

    LLValue *oldLength = DtoArrayLen(array);

    // Do not move exp->toElem call after creating _d_arrayappendcTX,
    // otherwise a ~= a[$-i] won't work correctly
    DValue *expVal = toElem(exp);

    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_arrayappendcTX");
    LLValue* args[] = {
        DtoTypeInfoOf(arrayType),
        DtoBitCast(array->getLVal(), fn->getFunctionType()->getParamType(1)),
        DtoConstSize_t(1)
    };

    LLValue* appendedArray = gIR->CreateCallOrInvoke(fn, args, ".appendedArray").getInstruction();
    appendedArray = DtoAggrPaint(appendedArray, DtoType(arrayType));

    LLValue* val = DtoArrayPtr(array);
    val = DtoGEP1(val, oldLength, "lastElem");
    DtoAssign(loc, new DVarValue(arrayType->nextOf(), val), expVal);
    callPostblit(loc, exp, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

DSliceValue* DtoCatAssignArray(Loc& loc, DValue* arr, Expression* exp)
{
    IF_LOG Logger::println("DtoCatAssignArray");
    LOG_SCOPE;
    Type *arrayType = arr->getType();

    // Prepare arguments
    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_arrayappendT");
    LLSmallVector<LLValue*,3> args;
    // TypeInfo ti
    args.push_back(DtoTypeInfoOf(arrayType));
    // byte[] *px
    args.push_back(DtoBitCast(arr->getLVal(), fn->getFunctionType()->getParamType(1)));
    // byte[] y
    LLValue *y = DtoSlice(toElem(exp));
    y = DtoAggrPaint(y, fn->getFunctionType()->getParamType(2));
    args.push_back(y);

    // Call _d_arrayappendT
    LLValue* newArray = gIR->CreateCallOrInvoke(fn, args, ".appendedArray").getInstruction();

    return getSlice(arrayType, newArray);
}

//////////////////////////////////////////////////////////////////////////////////////////

DSliceValue* DtoCatArrays(Loc& loc, Type* arrayType, Expression* exp1, Expression* exp2)
{
    IF_LOG Logger::println("DtoCatAssignArray");
    LOG_SCOPE;

    llvm::SmallVector<llvm::Value*, 3> args;
    LLFunction* fn = 0;

    if (exp1->op == TOKcat)
    { // handle multiple concat
        fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_arraycatnTX");

        // Create array of slices
        typedef llvm::SmallVector<llvm::Value*, 16> ArgVector;
        ArgVector arrs;
        arrs.push_back(DtoSlicePtr(toElem(exp2)));
        CatExp *ce = static_cast<CatExp*>(exp1);
        do
        {
            arrs.push_back(DtoSlicePtr(toElem(ce->e2)));
            ce = static_cast<CatExp *>(ce->e1);
        } while (ce->op == TOKcat);
        arrs.push_back(DtoSlicePtr(toElem(ce)));

        // Create static array from slices
        LLPointerType* ptrarraytype = isaPointer(arrs[0]->getType());
        assert(ptrarraytype && "Expected pointer type");
        LLStructType* arraytype = isaStruct(ptrarraytype->getElementType());
        assert(arraytype && "Expected struct type");
        LLArrayType* type = LLArrayType::get(arraytype, arrs.size());
        LLValue* array = DtoRawAlloca(type, 0, ".slicearray");
        unsigned int i = 0;
        for (ArgVector::reverse_iterator I = arrs.rbegin(), E = arrs.rend(); I != E; ++I)
        {
            LLValue* v = DtoLoad(DtoBitCast(*I, ptrarraytype));
            DtoStore(v, DtoGEPi(array, 0, i++, ".slice"));
        }

        LLStructType* type2 = DtoArrayType(arraytype);
        LLValue* array2 = DtoRawAlloca(type2, 0, ".array");
        DtoStore(DtoConstSize_t(arrs.size()), DtoGEPi(array2, 0, 0, ".len"));
        DtoStore(DtoBitCast(array, ptrarraytype), DtoGEPi(array2, 0, 1, ".ptr"));
        LLValue* val = DtoLoad(DtoBitCast(array2, getPtrToType(DtoArrayType(DtoArrayType(LLType::getInt8Ty(gIR->context()))))));

        // TypeInfo ti
        args.push_back(DtoTypeInfoOf(arrayType));
        // byte[][] arrs
        args.push_back(val);
    }
    else
    {
        fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_arraycatT");

        // TypeInfo ti
        args.push_back(DtoTypeInfoOf(arrayType));
        // byte[] x
        LLValue *val = DtoLoad(DtoSlicePtr(toElem(exp1)));
        val = DtoAggrPaint(val, fn->getFunctionType()->getParamType(1));
        args.push_back(val);
        // byte[] y
        val = DtoLoad(DtoSlicePtr(toElem(exp2)));
        val = DtoAggrPaint(val, fn->getFunctionType()->getParamType(2));
        args.push_back(val);
    }

    LLValue *newArray = gIR->CreateCallOrInvoke(fn, args, ".appendedArray").getInstruction();
    return getSlice(arrayType, newArray);
}

//////////////////////////////////////////////////////////////////////////////////////////

DSliceValue* DtoAppendDChar(Loc& loc, DValue* arr, Expression* exp, const char *func)
{
    Type *arrayType = arr->getType();
    DValue* valueToAppend = toElem(exp);

    // Prepare arguments
    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, func);
    LLValue* args[] = {
        // ref string x
        DtoBitCast(arr->getLVal(), fn->getFunctionType()->getParamType(0)),
        // dchar c
        DtoBitCast(valueToAppend->getRVal(), fn->getFunctionType()->getParamType(1))
    };

    // Call function
    LLValue* newArray = gIR->CreateCallOrInvoke(fn, args, ".appendedArray").getInstruction();

    return getSlice(arrayType, newArray);
}

//////////////////////////////////////////////////////////////////////////////////////////

DSliceValue* DtoAppendDCharToString(Loc& loc, DValue* arr, Expression* exp)
{
    IF_LOG Logger::println("DtoAppendDCharToString");
    LOG_SCOPE;
    return DtoAppendDChar(loc, arr, exp, "_d_arrayappendcd");
}

//////////////////////////////////////////////////////////////////////////////////////////

DSliceValue* DtoAppendDCharToUnicodeString(Loc& loc, DValue* arr, Expression* exp)
{
    IF_LOG Logger::println("DtoAppendDCharToUnicodeString");
    LOG_SCOPE;
    return DtoAppendDChar(loc, arr, exp, "_d_arrayappendwd");
}

//////////////////////////////////////////////////////////////////////////////////////////
// helper for eq and cmp
static LLValue* DtoArrayEqCmp_impl(Loc& loc, const char* func, DValue* l, DValue* r, bool useti)
{
    IF_LOG Logger::println("comparing arrays");
    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, func);
    assert(fn);

    // find common dynamic array type
    Type* commonType = l->getType()->toBasetype()->nextOf()->arrayOf();

    // cast static arrays to dynamic ones, this turns them into DSliceValues
    Logger::println("casting to dynamic arrays");
    l = DtoCastArray(loc, l, commonType);
    r = DtoCastArray(loc, r, commonType);

    LLSmallVector<LLValue*, 3> args;

    // get values, reinterpret cast to void[]
    args.push_back(DtoAggrPaint(l->getRVal(), DtoArrayType(LLType::getInt8Ty(gIR->context()))));
    args.push_back(DtoAggrPaint(r->getRVal(), DtoArrayType(LLType::getInt8Ty(gIR->context()))));

    // pass array typeinfo ?
    if (useti) {
        Type* t = l->getType();
        LLValue* tival = DtoTypeInfoOf(t);
        args.push_back(DtoBitCast(tival, fn->getFunctionType()->getParamType(2)));
    }

    LLCallSite call = gIR->CreateCallOrInvoke(fn, args);

    return call.getInstruction();
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayEquals(Loc& loc, TOK op, DValue* l, DValue* r)
{
    LLValue* res = DtoArrayEqCmp_impl(loc, _adEq, l, r, true);
    res = gIR->ir->CreateICmpNE(res, DtoConstInt(0));
    if (op == TOKnotequal)
        res = gIR->ir->CreateNot(res);

    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayCompare(Loc& loc, TOK op, DValue* l, DValue* r)
{
    LLValue* res = 0;
    llvm::ICmpInst::Predicate cmpop;
    tokToIcmpPred(op, false, &cmpop, &res);

    if (!res)
    {
        Type* t = l->getType()->toBasetype()->nextOf()->toBasetype();
        if (t->ty == Tchar)
            res = DtoArrayEqCmp_impl(loc, "_adCmpChar", l, r, false);
        else
            res = DtoArrayEqCmp_impl(loc, _adCmp, l, r, true);
        res = gIR->ir->CreateICmp(cmpop, res, DtoConstInt(0));
    }

    assert(res);
    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayCastLength(Loc& loc, LLValue* len, LLType* elemty, LLType* newelemty)
{
    IF_LOG Logger::println("DtoArrayCastLength");
    LOG_SCOPE;

    assert(len);
    assert(elemty);
    assert(newelemty);

    size_t esz = getTypePaddedSize(elemty);
    size_t nsz = getTypePaddedSize(newelemty);
    if (esz == nsz)
        return len;

    LLValue* args[] = {
        len,
        LLConstantInt::get(DtoSize_t(), esz, false),
        LLConstantInt::get(DtoSize_t(), nsz, false)
    };

    LLFunction* fn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_array_cast_len");
    return gIR->CreateCallOrInvoke(fn, args).getInstruction();
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoDynArrayIs(TOK op, DValue* l, DValue* r)
{
    LLValue *len1, *ptr1, *len2, *ptr2;

    assert(l);
    assert(r);

    // compare lengths
    len1 = DtoArrayLen(l);
    len2 = DtoArrayLen(r);
    LLValue* b1 = gIR->ir->CreateICmpEQ(len1,len2);

    // compare pointers
    ptr1 = DtoArrayPtr(l);
    ptr2 = DtoArrayPtr(r);
    LLValue* b2 = gIR->ir->CreateICmpEQ(ptr1,ptr2);

    // combine
    LLValue* res = gIR->ir->CreateAnd(b1,b2);

    // return result
    return (op == TOKnotidentity) ? gIR->ir->CreateNot(res) : res;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayLen(DValue* v)
{
    IF_LOG Logger::println("DtoArrayLen");
    LOG_SCOPE;

    Type* t = v->getType()->toBasetype();
    if (t->ty == Tarray) {
        if (DSliceValue* s = v->isSlice())
            return s->len;
        else if (v->isNull())
            return DtoConstSize_t(0);
        else if (v->isLVal())
            return DtoLoad(DtoGEPi(v->getLVal(), 0,0), ".len");
        return gIR->ir->CreateExtractValue(v->getRVal(), 0, ".len");
    }
    else if (t->ty == Tsarray) {
        assert(!v->isSlice());
        assert(!v->isNull());
        assert(v->type->toBasetype()->ty == Tsarray);
        TypeSArray *sarray = static_cast<TypeSArray*>(v->type->toBasetype());
        return DtoConstSize_t(sarray->dim->toUInteger());
    }
    llvm_unreachable("unsupported array for len");
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayPtr(DValue* v)
{
    IF_LOG Logger::println("DtoArrayPtr");
    LOG_SCOPE;

    Type* t = v->getType()->toBasetype();
    if (t->ty == Tarray) {
        if (DSliceValue* s = v->isSlice())
            return s->ptr;
        else if (v->isNull())
            return getNullPtr(getPtrToType(i1ToI8(DtoType(t->nextOf()))));
        else if (v->isLVal())
            return DtoLoad(DtoGEPi(v->getLVal(), 0,1), ".ptr");
        return gIR->ir->CreateExtractValue(v->getRVal(), 1, ".ptr");
    }
    else if (t->ty == Tsarray) {
        assert(!v->isSlice());
        assert(!v->isNull());
        return DtoGEPi(v->getRVal(), 0, 0, "sarrayptr");
    }

    llvm_unreachable("Unexpected array type.");
}

//////////////////////////////////////////////////////////////////////////////////////////
DValue* DtoCastArray(Loc& loc, DValue* u, Type* to)
{
    IF_LOG Logger::println("DtoCastArray");
    LOG_SCOPE;

    LLType* tolltype = DtoType(to);

    Type* totype = to->toBasetype();
    Type* fromtype = u->getType()->toBasetype();
    if (fromtype->ty != Tarray && fromtype->ty != Tsarray) {
        error(loc, "can't cast %s to %s", u->getType()->toChars(), to->toChars());
        fatal();
    }

    LLValue* rval;
    LLValue* rval2;
    bool isslice = false;

    IF_LOG Logger::cout() << "from array or sarray" << '\n';

    if (totype->ty == Tpointer) {
        IF_LOG Logger::cout() << "to pointer" << '\n';
        rval = DtoArrayPtr(u);
        if (rval->getType() != tolltype)
            rval = gIR->ir->CreateBitCast(rval, tolltype);
    }
    else if (totype->ty == Tarray) {
        IF_LOG Logger::cout() << "to array" << '\n';

        LLType* ptrty = DtoArrayType(totype)->getContainedType(1);
        LLType* ety = voidToI8(DtoType(fromtype->nextOf()));

        if (fromtype->ty == Tsarray) {
            LLValue* uval = u->getRVal();

            IF_LOG Logger::cout() << "uvalTy = " << *uval->getType() << '\n';

            assert(isaPointer(uval->getType()));
            LLArrayType* arrty = isaArray(uval->getType()->getContainedType(0));

            if(arrty->getNumElements()*fromtype->nextOf()->size() % totype->nextOf()->size() != 0)
            {
                error(loc, "invalid cast from '%s' to '%s', the element sizes don't line up", fromtype->toChars(), totype->toChars());
                fatal();
            }

            uinteger_t len = static_cast<TypeSArray*>(fromtype)->dim->toUInteger();
            rval2 = LLConstantInt::get(DtoSize_t(), len, false);
            if (fromtype->nextOf()->size() != totype->nextOf()->size())
                rval2 = DtoArrayCastLength(loc, rval2, ety, ptrty->getContainedType(0));
            rval = DtoBitCast(uval, ptrty);
        }
        else {
            rval2 = DtoArrayLen(u);
            if (fromtype->nextOf()->size() != totype->nextOf()->size())
                rval2 = DtoArrayCastLength(loc, rval2, ety, ptrty->getContainedType(0));

            rval = DtoArrayPtr(u);
            rval = DtoBitCast(rval, ptrty);
        }
        isslice = true;
    }
    else if (totype->ty == Tsarray) {
        IF_LOG Logger::cout() << "to sarray" << '\n';

        size_t tosize = static_cast<TypeSArray*>(totype)->dim->toInteger();

        if (fromtype->ty == Tsarray) {
            LLValue* uval = u->getRVal();

            IF_LOG Logger::cout() << "uvalTy = " << *uval->getType() << '\n';

            assert(isaPointer(uval->getType()));

            /*LLArrayType* arrty = isaArray(uval->getType()->getContainedType(0));
            if(arrty->getNumElements()*fromtype->nextOf()->size() != tosize*totype->nextOf()->size())
            {
                error(loc, "invalid cast from '%s' to '%s', the sizes are not the same", fromtype->toChars(), totype->toChars());
                fatal();
            }*/

            rval = DtoBitCast(uval, getPtrToType(tolltype));
        }
        else {
            size_t i = (tosize * totype->nextOf()->size() - 1) / fromtype->nextOf()->size();
            DConstValue index(Type::tsize_t, DtoConstSize_t(i));
            DtoArrayBoundsCheck(loc, u, &index);

            rval = DtoArrayPtr(u);
            rval = DtoBitCast(rval, getPtrToType(tolltype));
        }
    }
    else if (totype->ty == Tbool) {
        // return (arr.ptr !is null)
        LLValue* ptr = DtoArrayPtr(u);
        LLConstant* nul = getNullPtr(ptr->getType());
        rval = gIR->ir->CreateICmpNE(ptr, nul);
    }
    else {
        rval = DtoArrayPtr(u);
        rval = DtoBitCast(rval, getPtrToType(tolltype));
        if (totype->ty != Tstruct)
            rval = DtoLoad(rval);
    }

    if (isslice) {
        Logger::println("isslice");
        return new DSliceValue(to, rval2, rval);
    }

    return new DImValue(to, rval);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoArrayBoundsCheck(Loc& loc, DValue* arr, DValue* index, DValue* lowerBound)
{
    Type* arrty = arr->getType()->toBasetype();
    assert((arrty->ty == Tsarray || arrty->ty == Tarray || arrty->ty == Tpointer) &&
        "Can only array bounds check for static or dynamic arrays");

    // We do not check if the bounds check can be omitted. This is the
    // responsibility of the caller and performed in IndexExp::toElem().

    // runtime check

    bool lengthUnknown = arrty->ty == Tpointer;

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* failbb = llvm::BasicBlock::Create(gIR->context(), "arrayboundscheckfail", gIR->topfunc(), oldend);
    llvm::BasicBlock* okbb = llvm::BasicBlock::Create(gIR->context(), "arrayboundsok", gIR->topfunc(), oldend);
    LLValue* cond = 0;

    if (!lengthUnknown) {
        // if lowerBound is not NULL, we're checking slice
        llvm::ICmpInst::Predicate cmpop = lowerBound ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_ULT;
        // check for upper bound
        cond = gIR->ir->CreateICmp(cmpop, index->getRVal(), DtoArrayLen(arr), "boundscheck");
    }

    if (!lowerBound) {
        assert(cond);
        gIR->ir->CreateCondBr(cond, okbb, failbb);
    } else {
        if (!lengthUnknown) {
            llvm::BasicBlock* locheckbb = llvm::BasicBlock::Create(gIR->context(), "arrayboundschecklowerbound", gIR->topfunc(), oldend);
            gIR->ir->CreateCondBr(cond, locheckbb, failbb);
            gIR->scope() = IRScope(locheckbb, failbb);
        }
        // check for lower bound
        cond = gIR->ir->CreateICmp(llvm::ICmpInst::ICMP_ULE, lowerBound->getRVal(), index->getRVal(), "boundscheck");
        gIR->ir->CreateCondBr(cond, okbb, failbb);
    }

    // set up failbb to call the array bounds error runtime function

    gIR->scope() = IRScope(failbb, okbb);

    std::vector<LLValue*> args;

    // file param
    Module* funcmodule = gIR->func()->decl->getModule();
    args.push_back(DtoModuleFileName(funcmodule, loc));

    // line param
    LLConstant* c = DtoConstUint(loc.linnum);
    args.push_back(c);

    // call
    llvm::Function* errorfn = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_d_arraybounds");
    gIR->CreateCallOrInvoke(errorfn, args);

    // the function does not return
    gIR->ir->CreateUnreachable();

    // if ok, proceed in okbb
    gIR->scope() = IRScope(okbb, oldend);
}
