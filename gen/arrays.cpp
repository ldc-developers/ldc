#include "gen/llvm.h"

#include "mtype.h"
#include "module.h"
#include "dsymbol.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/runtime.h"
#include "gen/logger.h"
#include "gen/dvalue.h"
#include "ir/irmodule.h"

#include "gen/cl_options.h"

//////////////////////////////////////////////////////////////////////////////////////////

const LLStructType* DtoArrayType(Type* arrayTy)
{
    assert(arrayTy->nextOf());
    const LLType* elemty = DtoType(arrayTy->nextOf());
    if (elemty == LLType::getVoidTy(gIR->context()))
        elemty = LLType::getInt8Ty(gIR->context());
    return LLStructType::get(gIR->context(), DtoSize_t(), getPtrToType(elemty), NULL);
}

const LLStructType* DtoArrayType(const LLType* t)
{
    return LLStructType::get(gIR->context(), DtoSize_t(), getPtrToType(t), NULL);
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLArrayType* DtoStaticArrayType(Type* t)
{
    t = t->toBasetype();
    assert(t->ty == Tsarray);
    TypeSArray* tsa = (TypeSArray*)t;
    Type* tnext = tsa->nextOf();

    const LLType* elemty = DtoType(tnext);
    if (elemty == LLType::getVoidTy(gIR->context()))
        elemty = LLType::getInt8Ty(gIR->context());

    return LLArrayType::get(elemty, tsa->dim->toUInteger());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoSetArrayToNull(LLValue* v)
{
    Logger::println("DtoSetArrayToNull");
    LOG_SCOPE;

    assert(isaPointer(v));
    const LLType* t = v->getType()->getContainedType(0);

    DtoStore(LLConstant::getNullValue(t), v);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoArrayInit(Loc& loc, DValue* array, DValue* value)
{
    Logger::println("DtoArrayInit");
    LOG_SCOPE;

    LLValue* dim = DtoArrayLen(array);
    LLValue* ptr = DtoArrayPtr(array);
    LLValue* val;

    // give slices and complex values storage (and thus an address to pass)
    if (value->isSlice())
    {
        val = DtoAlloca(value->getType(), ".tmpparam");
        DVarValue lval(value->getType(), val);
        DtoAssign(loc, &lval, value);
    }
    else
    {
        val = value->getRVal();
    }
    assert(val);

    // prepare runtime call
    LLSmallVector<LLValue*, 4> args;
    args.push_back(ptr);
    args.push_back(dim);
    args.push_back(val);

    // determine the right runtime function to call
    const char* funcname = NULL;
    Type* arrayelemty = array->getType()->nextOf()->toBasetype();
    Type* valuety = value->getType()->toBasetype();

    // lets first optimize all zero initializations down to a memset.
    // this simplifies codegen later on as llvm null's have no address!
    if (isaConstant(val) && isaConstant(val)->isNullValue())
    {
        size_t X = getTypePaddedSize(val->getType());
        LLValue* nbytes = gIR->ir->CreateMul(dim, DtoConstSize_t(X), ".nbytes");
        DtoMemSetZero(ptr, nbytes);
        return;
    }

    // if not a zero initializer, call the appropriate runtime function!
    switch (arrayelemty->ty)
    {
    case Tbool:
        val = gIR->ir->CreateZExt(val, LLType::getInt8Ty(gIR->context()), ".bool");
        // fall through

    case Tvoid:
    case Tchar:
    case Tint8:
    case Tuns8:
        Logger::println("Using memset for array init");
        DtoMemSet(ptr, val, dim);
        return;

    case Twchar:
    case Tint16:
    case Tuns16:
        funcname = "_d_array_init_i16";
        break;

    case Tdchar:
    case Tint32:
    case Tuns32:
        funcname = "_d_array_init_i32";
        break;

    case Tint64:
    case Tuns64:
        funcname = "_d_array_init_i64";
        break;

    case Tfloat32:
    case Timaginary32:
        funcname = "_d_array_init_float";
        break;

    case Tfloat64:
    case Timaginary64:
        funcname = "_d_array_init_double";
        break;

    case Tfloat80:
    case Timaginary80:
        funcname = "_d_array_init_real";
        break;

    case Tcomplex32:
        funcname = "_d_array_init_cfloat";
        break;

    case Tcomplex64:
        funcname = "_d_array_init_cdouble";
        break;

    case Tcomplex80:
        funcname = "_d_array_init_creal";
        break;

    case Tpointer:
    case Tclass:
        funcname = "_d_array_init_pointer";
        args[0] = DtoBitCast(args[0], getPtrToType(getVoidPtrType()));
        args[2] = DtoBitCast(args[2], getVoidPtrType());
        break;

    // this currently acts as a kind of fallback for all the bastards...
    // FIXME: this is probably too slow.
    case Tstruct:
    case Tdelegate:
    case Tarray:
    case Tsarray:
        funcname = "_d_array_init_mem";
        assert(arrayelemty == valuety && "ArrayInit doesn't work on elem-initialized static arrays");
        args[0] = DtoBitCast(args[0], getVoidPtrType());
        args[2] = DtoBitCast(args[2], getVoidPtrType());
        args.push_back(DtoConstSize_t(getTypePaddedSize(DtoTypeNotVoid(arrayelemty))));
        break;

    default:
        error("unhandled array init: %s = %s", array->getType()->toChars(), value->getType()->toChars());
        assert(0 && "unhandled array init");
    }

    if (Logger::enabled())
    {
        Logger::cout() << "ptr = " << *args[0] << std::endl;
        Logger::cout() << "dim = " << *args[1] << std::endl;
        Logger::cout() << "val = " << *args[2] << std::endl;
    }

    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, funcname);
    assert(fn);
    if (Logger::enabled())
        Logger::cout() << "calling array init function: " << *fn <<'\n';
    LLCallSite call = gIR->CreateCallOrInvoke(fn, args.begin(), args.end());
    call.setCallingConv(llvm::CallingConv::C);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoSetArray(LLValue* arr, LLValue* dim, LLValue* ptr)
{
    Logger::println("SetArray");
    assert(isaStruct(arr->getType()->getContainedType(0)));
    DtoStore(dim, DtoGEPi(arr,0,0));
    DtoStore(ptr, DtoGEPi(arr,0,1));
}

//////////////////////////////////////////////////////////////////////////////////////////

LLConstant* DtoConstArrayInitializer(ArrayInitializer* arrinit)
{
    Logger::println("DtoConstArrayInitializer: %s | %s", arrinit->toChars(), arrinit->type->toChars());
    LOG_SCOPE;

    assert(arrinit->value.dim == arrinit->index.dim);

    // get base array type
    Type* arrty = arrinit->type->toBasetype();
    size_t arrlen = arrinit->dim;

    // for statis arrays, dmd does not include any trailing default
    // initialized elements in the value/index lists
    if (arrty->ty == Tsarray)
    {
        TypeSArray* tsa = (TypeSArray*)arrty;
        arrlen = (size_t)tsa->dim->toInteger();
    }

    // make sure the number of initializers is sane
    if (arrinit->index.dim > arrlen || arrinit->dim > arrlen)
    {
        error(arrinit->loc, "too many initializers, %u, for array[%zu]", arrinit->index.dim, arrlen);
        fatal();
    }

    // get elem type
    Type* elemty = arrty->nextOf();
    const LLType* llelemty = DtoTypeNotVoid(elemty);

    // true if array elements differ in type, can happen with array of unions
    bool mismatch = false;

    // allocate room for initializers
    std::vector<LLConstant*> initvals(arrlen, NULL);

    // go through each initializer, they're not sorted by index by the frontend
    size_t j = 0;
    for (size_t i = 0; i < arrinit->index.dim; i++)
    {
        // get index
        Expression* idx = (Expression*)arrinit->index.data[i];

        // idx can be null, then it's just the next element
        if (idx)
            j = idx->toInteger();
        assert(j < arrlen);

        // get value
        Initializer* val = (Initializer*)arrinit->value.data[i];
        assert(val);

        // error check from dmd
        if (initvals[j] != NULL)
        {
            error(arrinit->loc, "duplicate initialization for index %zu", j);
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
        constarr = LLConstantStruct::get(gIR->context(), initvals, false); // FIXME should this pack?
    else
        constarr = LLConstantArray::get(LLArrayType::get(llelemty, arrlen), initvals);

//     std::cout << "constarr: " << *constarr << std::endl;

    // if the type is a static array, we're done
    if (arrty->ty == Tsarray)
        return constarr;

    // for dynamic array we need to make a global with the data, so we have a pointer for the dynamic array
    // Important: don't make the gvar constant, since this const initializer might
    // be used as an initializer for a static T[] - where modifying contents is allowed.
    LLGlobalVariable* gvar = new LLGlobalVariable(*gIR->module, constarr->getType(), false, LLGlobalValue::InternalLinkage, constarr, ".constarray");
    LLConstant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };

    LLConstant* gep = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);
    gep = llvm::ConstantExpr::getBitCast(gvar, getPtrToType(llelemty));

    return DtoConstSlice(DtoConstSize_t(arrlen),gep);
}

//////////////////////////////////////////////////////////////////////////////////////////
static LLValue* get_slice_ptr(DSliceValue* e, LLValue*& sz)
{
    assert(e->len != 0);
    const LLType* t = e->ptr->getType()->getContainedType(0);
    sz = gIR->ir->CreateMul(DtoConstSize_t(getTypePaddedSize(t)), e->len, "tmp");
    return DtoBitCast(e->ptr, getVoidPtrType());
}

void DtoArrayCopySlices(DSliceValue* dst, DSliceValue* src)
{
    Logger::println("ArrayCopySlices");

    LLValue *sz1,*sz2;
    LLValue* dstarr = get_slice_ptr(dst,sz1);
    LLValue* srcarr = get_slice_ptr(src,sz2);

    if (global.params.useAssert || global.params.useArrayBounds)
    {
        LLValue* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_array_slice_copy");
        gIR->CreateCallOrInvoke4(fn, dstarr, sz1, srcarr, sz2);
    }
    else
    {
        DtoMemCpy(dstarr, srcarr, sz1);
    }
}

void DtoArrayCopyToSlice(DSliceValue* dst, DValue* src)
{
    Logger::println("ArrayCopyToSlice");

    LLValue* sz1;
    LLValue* dstarr = get_slice_ptr(dst,sz1);

    LLValue* srcarr = DtoBitCast(DtoArrayPtr(src), getVoidPtrType());
    const LLType* arrayelemty = DtoTypeNotVoid(src->getType()->nextOf()->toBasetype());
    LLValue* sz2 = gIR->ir->CreateMul(DtoConstSize_t(getTypePaddedSize(arrayelemty)), DtoArrayLen(src), "tmp");

    if (global.params.useAssert || global.params.useArrayBounds)
    {
        LLValue* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_array_slice_copy");
        gIR->CreateCallOrInvoke4(fn, dstarr, sz1, srcarr, sz2);
    }
    else
    {
        DtoMemCpy(dstarr, srcarr, sz1);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoStaticArrayCopy(LLValue* dst, LLValue* src)
{
    Logger::println("StaticArrayCopy");

    size_t n = getTypePaddedSize(dst->getType()->getContainedType(0));
    DtoMemCpy(dst, src, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////
LLConstant* DtoConstSlice(LLConstant* dim, LLConstant* ptr)
{
    LLConstant* values[2] = { dim, ptr };
    return LLConstantStruct::get(gIR->context(), values, 2, false);
}

//////////////////////////////////////////////////////////////////////////////////////////
static bool isInitialized(Type* et) {
    // Strip static array types from element type
    Type* bt = et->toBasetype();
    while (bt->ty == Tsarray) {
        et = bt->nextOf();
        bt = et->toBasetype();
    }
    // If it's a typedef with "= void" initializer then don't initialize.
    if (et->ty == Ttypedef) {
        Logger::println("Typedef: %s", et->toChars());
        TypedefDeclaration* tdd = ((TypeTypedef*)et)->sym;
        if (tdd && tdd->init && tdd->init->isVoidInitializer())
            return false;
    }
    // Otherwise, it's always initialized.
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoNewDynArray(Loc& loc, Type* arrayType, DValue* dim, bool defaultInit)
{
    Logger::println("DtoNewDynArray : %s", arrayType->toChars());
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
    const char* fnname = defaultInit ? (zeroInit ? "_d_newarrayT" : "_d_newarrayiT") : "_d_newarrayvT";
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, fnname);

    // call allocator
    LLValue* newptr = gIR->CreateCallOrInvoke2(fn, arrayTypeInfo, arrayLen, ".gc_mem").getInstruction();

    // cast to wanted type
    const LLType* dstType = DtoType(arrayType)->getContainedType(1);
    if (newptr->getType() != dstType)
        newptr = DtoBitCast(newptr, dstType, ".gc_mem");

    if (Logger::enabled())
        Logger::cout() << "final ptr = " << *newptr << '\n';

    return new DSliceValue(arrayType, arrayLen, newptr);
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoNewMulDimDynArray(Loc& loc, Type* arrayType, DValue** dims, size_t ndims, bool defaultInit)
{
    Logger::println("DtoNewMulDimDynArray : %s", arrayType->toChars());
    LOG_SCOPE;

    // typeinfo arg
    LLValue* arrayTypeInfo = DtoTypeInfoOf(arrayType);

    // get value type
    Type* vtype = arrayType->toBasetype();
    for (size_t i=0; i<ndims; ++i)
        vtype = vtype->nextOf();

    // get runtime function
    bool zeroInit = vtype->isZeroInit();
    if (defaultInit && !isInitialized(vtype))
        defaultInit = false;
    const char* fnname = defaultInit ? (zeroInit ? "_d_newarraymT" : "_d_newarraymiT") : "_d_newarraymvT";
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, fnname);

    // build dims
    LLValue* dimsArg = DtoArrayAlloca(Type::tsize_t, ndims, ".newdims");
    LLValue* firstDim = NULL;
    for (size_t i=0; i<ndims; ++i)
    {
        LLValue* dim = dims[i]->getRVal();
        if (!firstDim) firstDim = dim;
        DtoStore(dim, DtoGEPi1(dimsArg, i));
    }

    // call allocator
    LLValue* newptr = gIR->CreateCallOrInvoke3(fn, arrayTypeInfo, DtoConstSize_t(ndims), dimsArg, ".gc_mem").getInstruction();

    // cast to wanted type
    const LLType* dstType = DtoType(arrayType)->getContainedType(1);
    if (newptr->getType() != dstType)
        newptr = DtoBitCast(newptr, dstType, ".gc_mem");

    if (Logger::enabled())
        Logger::cout() << "final ptr = " << *newptr << '\n';

    assert(firstDim);
    return new DSliceValue(arrayType, firstDim, newptr);
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoResizeDynArray(Type* arrayType, DValue* array, DValue* newdim)
{
    Logger::println("DtoResizeDynArray : %s", arrayType->toChars());
    LOG_SCOPE;

    assert(array);
    assert(newdim);
    assert(arrayType);
    assert(arrayType->toBasetype()->ty == Tarray);

    // decide on what runtime function to call based on whether the type is zero initialized
    bool zeroInit = arrayType->toBasetype()->nextOf()->isZeroInit();

    // call runtime
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, zeroInit ? "_d_arraysetlengthT" : "_d_arraysetlengthiT" );

    LLSmallVector<LLValue*,4> args;
    args.push_back(DtoTypeInfoOf(arrayType));
    args.push_back(newdim->getRVal());
    args.push_back(DtoArrayLen(array));

    LLValue* arrPtr = DtoArrayPtr(array);
    if (Logger::enabled())
        Logger::cout() << "arrPtr = " << *arrPtr << '\n';
    args.push_back(DtoBitCast(arrPtr, fn->getFunctionType()->getParamType(3), "tmp"));

    LLValue* newptr = gIR->CreateCallOrInvoke(fn, args.begin(), args.end(), ".gc_mem").getInstruction();
    if (newptr->getType() != arrPtr->getType())
        newptr = DtoBitCast(newptr, arrPtr->getType(), ".gc_mem");

    return new DSliceValue(arrayType, newdim->getRVal(), newptr);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoCatAssignElement(Loc& loc, Type* arrayType, DValue* array, Expression* exp)
{
    Logger::println("DtoCatAssignElement");
    LOG_SCOPE;

    assert(array);

    LLValue *valueToAppend = makeLValue(loc, exp->toElem(gIR));

    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_arrayappendcT");
    LLSmallVector<LLValue*,3> args;
    args.push_back(DtoTypeInfoOf(arrayType));
    args.push_back(DtoBitCast(array->getLVal(), getVoidPtrType()));
    args.push_back(DtoBitCast(valueToAppend, getVoidPtrType()));

    gIR->CreateCallOrInvoke(fn, args.begin(), args.end(), ".appendedArray");
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoCatAssignArray(DValue* arr, Expression* exp)
{
    Logger::println("DtoCatAssignArray");
    LOG_SCOPE;

    DValue* e = exp->toElem(gIR);

    llvm::Value *len1, *len2, *src1, *src2, *res;

    len1 = DtoArrayLen(arr);
    len2 = DtoArrayLen(e);
    res = gIR->ir->CreateAdd(len1,len2,"tmp");

    DValue* newdim = new DImValue(Type::tsize_t, res);
    DSliceValue* slice = DtoResizeDynArray(arr->getType(), arr, newdim);

    src1 = slice->ptr;
    src2 = DtoArrayPtr(e);

    // advance ptr
    src1 = gIR->ir->CreateGEP(src1,len1,"tmp");

    // memcpy
    LLValue* elemSize = DtoConstSize_t(getTypePaddedSize(src2->getType()->getContainedType(0)));
    LLValue* bytelen = gIR->ir->CreateMul(len2, elemSize, "tmp");
    DtoMemCpy(src1,src2,bytelen);

    return slice;
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoCatArrays(Type* type, Expression* exp1, Expression* exp2)
{
    Logger::println("DtoCatArrays");
    LOG_SCOPE;

    Type* t1 = exp1->type->toBasetype();
    Type* t2 = exp2->type->toBasetype();

    assert(t1->ty == Tarray || t1->ty == Tsarray);
    assert(t2->ty == Tarray || t2->ty == Tsarray);

    DValue* e1 = exp1->toElem(gIR);
    DValue* e2 = exp2->toElem(gIR);

    llvm::Value *len1, *len2, *src1, *src2, *res;

    len1 = DtoArrayLen(e1);
    len2 = DtoArrayLen(e2);
    res = gIR->ir->CreateAdd(len1,len2,"tmp");

    DValue* lenval = new DImValue(Type::tsize_t, res);
    DSliceValue* slice = DtoNewDynArray(exp1->loc, type, lenval, false);
    LLValue* mem = slice->ptr;

    src1 = DtoArrayPtr(e1);
    src2 = DtoArrayPtr(e2);

    // first memcpy
    LLValue* elemSize = DtoConstSize_t(getTypePaddedSize(src1->getType()->getContainedType(0)));
    LLValue* bytelen = gIR->ir->CreateMul(len1, elemSize, "tmp");
    DtoMemCpy(mem,src1,bytelen);

    // second memcpy
    mem = gIR->ir->CreateGEP(mem,len1,"tmp");
    bytelen = gIR->ir->CreateMul(len2, elemSize, "tmp");
    DtoMemCpy(mem,src2,bytelen);

    return slice;
}

//////////////////////////////////////////////////////////////////////////////////////////
DSliceValue* DtoCatArrayElement(Type* type, Expression* exp1, Expression* exp2)
{
    Logger::println("DtoCatArrayElement");
    LOG_SCOPE;

    Type* t1 = exp1->type->toBasetype();
    Type* t2 = exp2->type->toBasetype();

    DValue* e1 = exp1->toElem(gIR);
    DValue* e2 = exp2->toElem(gIR);

    llvm::Value *len1, *src1, *res;

    // handle prefix case, eg. int~int[]
    if (t2->nextOf() && t1 == t2->nextOf()->toBasetype())
    {
        len1 = DtoArrayLen(e2);
        res = gIR->ir->CreateAdd(len1,DtoConstSize_t(1),"tmp");

        DValue* lenval = new DImValue(Type::tsize_t, res);
        DSliceValue* slice = DtoNewDynArray(exp1->loc, type, lenval, false);
        LLValue* mem = slice->ptr;

        DVarValue* memval = new DVarValue(e1->getType(), mem);
        DtoAssign(exp1->loc, memval, e1);

        src1 = DtoArrayPtr(e2);

        mem = gIR->ir->CreateGEP(mem,DtoConstSize_t(1),"tmp");

        LLValue* elemSize = DtoConstSize_t(getTypePaddedSize(src1->getType()->getContainedType(0)));
        LLValue* bytelen = gIR->ir->CreateMul(len1, elemSize, "tmp");
        DtoMemCpy(mem,src1,bytelen);


        return slice;
    }
    // handle suffix case, eg. int[]~int
    else
    {
        len1 = DtoArrayLen(e1);
        res = gIR->ir->CreateAdd(len1,DtoConstSize_t(1),"tmp");

        DValue* lenval = new DImValue(Type::tsize_t, res);
        DSliceValue* slice = DtoNewDynArray(exp1->loc, type, lenval, false);
        LLValue* mem = slice->ptr;

        src1 = DtoArrayPtr(e1);

        LLValue* elemSize = DtoConstSize_t(getTypePaddedSize(src1->getType()->getContainedType(0)));
        LLValue* bytelen = gIR->ir->CreateMul(len1, elemSize, "tmp");
        DtoMemCpy(mem,src1,bytelen);

        mem = gIR->ir->CreateGEP(mem,len1,"tmp");
        DVarValue* memval = new DVarValue(e2->getType(), mem);
        DtoAssign(exp1->loc, memval, e2);

        return slice;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// helper for eq and cmp
static LLValue* DtoArrayEqCmp_impl(Loc& loc, const char* func, DValue* l, DValue* r, bool useti)
{
    Logger::println("comparing arrays");
    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, func);
    assert(fn);

    // find common dynamic array type
    Type* commonType = l->getType()->toBasetype()->nextOf()->arrayOf();

    // cast static arrays to dynamic ones, this turns them into DSliceValues
    Logger::println("casting to dynamic arrays");
    l = DtoCastArray(loc, l, commonType);
    r = DtoCastArray(loc, r, commonType);

    LLValue* lmem;
    LLValue* rmem;
    LLSmallVector<LLValue*, 3> args;

    // get values, reinterpret cast to void[]
    lmem = DtoAggrPaint(l->getRVal(), DtoArrayType(LLType::getInt8Ty(gIR->context())));
    args.push_back(lmem);

    rmem = DtoAggrPaint(r->getRVal(), DtoArrayType(LLType::getInt8Ty(gIR->context())));
    args.push_back(rmem);

    // pass array typeinfo ?
    if (useti) {
        Type* t = l->getType();
        LLValue* tival = DtoTypeInfoOf(t);
        // DtoTypeInfoOf only does declare, not enough in this case :/
        t->vtinfo->codegen(Type::sir);

#if 0
        if (Logger::enabled())
            Logger::cout() << "typeinfo decl: " << *tival << '\n';
#endif

        args.push_back(DtoBitCast(tival, fn->getFunctionType()->getParamType(2)));
    }

    LLCallSite call = gIR->CreateCallOrInvoke(fn, args.begin(), args.end(), "tmp");

    return call.getInstruction();
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayEquals(Loc& loc, TOK op, DValue* l, DValue* r)
{
    LLValue* res = DtoArrayEqCmp_impl(loc, "_adEq", l, r, true);
    res = gIR->ir->CreateICmpNE(res, DtoConstInt(0), "tmp");
    if (op == TOKnotequal)
        res = gIR->ir->CreateNot(res, "tmp");

    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayCompare(Loc& loc, TOK op, DValue* l, DValue* r)
{
    LLValue* res = 0;

    llvm::ICmpInst::Predicate cmpop;
    bool skip = false;

    switch(op)
    {
    case TOKlt:
    case TOKul:
        cmpop = llvm::ICmpInst::ICMP_SLT;
        break;
    case TOKle:
    case TOKule:
        cmpop = llvm::ICmpInst::ICMP_SLE;
        break;
    case TOKgt:
    case TOKug:
        cmpop = llvm::ICmpInst::ICMP_SGT;
        break;
    case TOKge:
    case TOKuge:
        cmpop = llvm::ICmpInst::ICMP_SGE;
        break;
    case TOKue:
        cmpop = llvm::ICmpInst::ICMP_EQ;
        break;
    case TOKlg:
        cmpop = llvm::ICmpInst::ICMP_NE;
        break;
    case TOKleg:
        skip = true;
        res = LLConstantInt::getTrue(gIR->context());
        break;
    case TOKunord:
        skip = true;
        res = LLConstantInt::getFalse(gIR->context());
        break;

    default:
        assert(0);
    }

    if (!skip)
    {
        Type* t = l->getType()->toBasetype()->nextOf()->toBasetype();
        if (t->ty == Tchar)
            res = DtoArrayEqCmp_impl(loc, "_adCmpChar", l, r, false);
        else
            res = DtoArrayEqCmp_impl(loc, "_adCmp", l, r, true);
        res = gIR->ir->CreateICmp(cmpop, res, DtoConstInt(0), "tmp");
    }

    assert(res);
    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayCastLength(LLValue* len, const LLType* elemty, const LLType* newelemty)
{
    Logger::println("DtoArrayCastLength");
    LOG_SCOPE;

    assert(len);
    assert(elemty);
    assert(newelemty);

    size_t esz = getTypePaddedSize(elemty);
    size_t nsz = getTypePaddedSize(newelemty);
    if (esz == nsz)
        return len;

    LLSmallVector<LLValue*, 3> args;
    args.push_back(len);
    args.push_back(LLConstantInt::get(DtoSize_t(), esz, false));
    args.push_back(LLConstantInt::get(DtoSize_t(), nsz, false));

    LLFunction* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_array_cast_len");
    return gIR->CreateCallOrInvoke(fn, args.begin(), args.end(), "tmp").getInstruction();
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
    LLValue* b1 = gIR->ir->CreateICmpEQ(len1,len2,"tmp");

    // compare pointers
    ptr1 = DtoArrayPtr(l);
    ptr2 = DtoArrayPtr(r);
    LLValue* b2 = gIR->ir->CreateICmpEQ(ptr1,ptr2,"tmp");

    // combine
    LLValue* res = gIR->ir->CreateAnd(b1,b2,"tmp");

    // return result
    return (op == TOKnotidentity) ? gIR->ir->CreateNot(res) : res;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayLen(DValue* v)
{
    Logger::println("DtoArrayLen");
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
        LLValue* rv = v->getRVal();
        const LLArrayType* t = isaArray(rv->getType()->getContainedType(0));
        assert(t);
        return DtoConstSize_t(t->getNumElements());
    }
    assert(0 && "unsupported array for len");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
LLValue* DtoArrayPtr(DValue* v)
{
    Logger::println("DtoArrayPtr");
    LOG_SCOPE;

    Type* t = v->getType()->toBasetype();
    if (t->ty == Tarray) {
        if (DSliceValue* s = v->isSlice())
            return s->ptr;
        else if (v->isNull())
            return getNullPtr(getPtrToType(DtoType(t->nextOf())));
        else if (v->isLVal())
            return DtoLoad(DtoGEPi(v->getLVal(), 0,1), ".ptr");
        return gIR->ir->CreateExtractValue(v->getRVal(), 1, ".ptr");
    }
    else if (t->ty == Tsarray) {
        assert(!v->isSlice());
        assert(!v->isNull());
        return DtoGEPi(v->getRVal(), 0,0);
    }
    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
DValue* DtoCastArray(Loc& loc, DValue* u, Type* to)
{
    Logger::println("DtoCastArray");
    LOG_SCOPE;

    const LLType* tolltype = DtoType(to);

    Type* totype = to->toBasetype();
    Type* fromtype = u->getType()->toBasetype();
    if (fromtype->ty != Tarray && fromtype->ty != Tsarray) {
        error(loc, "can't cast %s to %s", u->getType()->toChars(), to->toChars());
        fatal();
    }

    LLValue* rval;
    LLValue* rval2;
    bool isslice = false;

    if (Logger::enabled())
        Logger::cout() << "from array or sarray" << '\n';

    if (totype->ty == Tpointer) {
        if (Logger::enabled())
            Logger::cout() << "to pointer" << '\n';
        rval = DtoArrayPtr(u);
        if (rval->getType() != tolltype)
            rval = gIR->ir->CreateBitCast(rval, tolltype, "tmp");
    }
    else if (totype->ty == Tarray) {
        if (Logger::enabled())
            Logger::cout() << "to array" << '\n';

        const LLType* ptrty = DtoArrayType(totype)->getContainedType(1);
        const LLType* ety = DtoTypeNotVoid(fromtype->nextOf());

        if (fromtype->ty == Tsarray) {
            LLValue* uval = u->getRVal();

            if (Logger::enabled())
                Logger::cout() << "uvalTy = " << *uval->getType() << '\n';

            assert(isaPointer(uval->getType()));
            const LLArrayType* arrty = isaArray(uval->getType()->getContainedType(0));

            if(arrty->getNumElements()*fromtype->nextOf()->size() % totype->nextOf()->size() != 0)
            {
                error(loc, "invalid cast from '%s' to '%s', the element sizes don't line up", fromtype->toChars(), totype->toChars());
                fatal();
            }

            rval2 = LLConstantInt::get(DtoSize_t(), arrty->getNumElements(), false);
            if (fromtype->nextOf()->size() != totype->nextOf()->size())
                rval2 = DtoArrayCastLength(rval2, ety, ptrty->getContainedType(0));
            rval = DtoBitCast(uval, ptrty);
        }
        else {
            rval2 = DtoArrayLen(u);
            if (fromtype->nextOf()->size() != totype->nextOf()->size())
                rval2 = DtoArrayCastLength(rval2, ety, ptrty->getContainedType(0));

            rval = DtoArrayPtr(u);
            rval = DtoBitCast(rval, ptrty);
        }
        isslice = true;
    }
    else if (totype->ty == Tsarray) {
        if (Logger::enabled())
            Logger::cout() << "to sarray" << '\n';

        size_t tosize = ((TypeSArray*)totype)->dim->toInteger();

        if (fromtype->ty == Tsarray) {
            LLValue* uval = u->getRVal();

            if (Logger::enabled())
                Logger::cout() << "uvalTy = " << *uval->getType() << '\n';

            assert(isaPointer(uval->getType()));
            const LLArrayType* arrty = isaArray(uval->getType()->getContainedType(0));

            /*if(arrty->getNumElements()*fromtype->nextOf()->size() != tosize*totype->nextOf()->size())
            {
                error(loc, "invalid cast from '%s' to '%s', the sizes are not the same", fromtype->toChars(), totype->toChars());
                fatal();
            }*/

            rval = DtoBitCast(uval, getPtrToType(tolltype));
        }
        else {
            size_t i = (tosize * totype->nextOf()->size() - 1) / fromtype->nextOf()->size();
            DConstValue index(Type::tsize_t, DtoConstSize_t(i));
            DtoArrayBoundsCheck(loc, u, &index, false);

            rval = DtoArrayPtr(u);
            rval = DtoBitCast(rval, getPtrToType(tolltype));
        }
    }
    else if (totype->ty == Tbool) {
        // return (arr.ptr !is null)
        LLValue* ptr = DtoArrayPtr(u);
        LLConstant* nul = getNullPtr(ptr->getType());
        rval = gIR->ir->CreateICmpNE(ptr, nul, "tmp");
    }
    else {
        assert(0);
    }

    if (isslice) {
        Logger::println("isslice");
        return new DSliceValue(to, rval2, rval);
    }

    return new DImValue(to, rval);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoArrayBoundsCheck(Loc& loc, DValue* arr, DValue* index, bool isslice)
{
    Type* arrty = arr->getType()->toBasetype();
    assert((arrty->ty == Tsarray || arrty->ty == Tarray) &&
        "Can only array bounds check for static or dynamic arrays");

    // static arrays could get static checks for static indices
    // but shouldn't since it might be generic code that's never executed

    // runtime check

    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* failbb = llvm::BasicBlock::Create(gIR->context(), "arrayboundscheckfail", gIR->topfunc(), oldend);
    llvm::BasicBlock* okbb = llvm::BasicBlock::Create(gIR->context(), "arrayboundsok", gIR->topfunc(), oldend);

    llvm::ICmpInst::Predicate cmpop = isslice ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_ULT;
    LLValue* cond = gIR->ir->CreateICmp(cmpop, index->getRVal(), DtoArrayLen(arr), "boundscheck");
    gIR->ir->CreateCondBr(cond, okbb, failbb);

    // set up failbb to call the array bounds error runtime function

    gIR->scope() = IRScope(failbb, okbb);

    std::vector<LLValue*> args;

    // file param
    // we might be generating for an imported template function
    Module* funcmodule = gIR->func()->decl->getModule();
    const char* cur_file = funcmodule->srcfile->name->toChars();
    if (loc.filename && strcmp(loc.filename, cur_file) != 0)
    {
        args.push_back(DtoConstString(loc.filename));
    }
    else
    {
        IrModule* irmod = getIrModule(funcmodule);
        args.push_back(DtoLoad(irmod->fileName));
    }

    // line param
    LLConstant* c = DtoConstUint(loc.linnum);
    args.push_back(c);

    // call
    llvm::Function* errorfn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_array_bounds");
    gIR->CreateCallOrInvoke(errorfn, args.begin(), args.end());

    // the function does not return
    gIR->ir->CreateUnreachable();

    // if ok, proceed in okbb
    gIR->scope() = IRScope(okbb, oldend);
}
