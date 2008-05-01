#include "gen/llvm.h"

#include "mtype.h"
#include "dsymbol.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/arrays.h"
#include "gen/runtime.h"
#include "gen/logger.h"
#include "gen/dvalue.h"

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* DtoArrayType(Type* t)
{
    assert(t->next);
    const llvm::Type* at = DtoType(t->next);
    const llvm::Type* arrty;

    if (at == llvm::Type::VoidTy) {
        at = llvm::Type::Int8Ty;
    }
    arrty = getPtrToType(at);

    std::vector<const llvm::Type*> members;
    if (global.params.is64bit)
        members.push_back(llvm::Type::Int64Ty);
    else
        members.push_back(llvm::Type::Int32Ty);

    members.push_back(arrty);

    return llvm::StructType::get(members);
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::ArrayType* DtoStaticArrayType(Type* t)
{
    if (t->llvmType)
        return isaArray(t->llvmType->get());

    assert(t->ty == Tsarray);
    assert(t->next);

    const llvm::Type* at = DtoType(t->next);

    TypeSArray* tsa = (TypeSArray*)t;
    assert(tsa->dim->type->isintegral());
    const llvm::ArrayType* arrty = llvm::ArrayType::get(at,tsa->dim->toUInteger());

    assert(!tsa->llvmType);
    tsa->llvmType = new llvm::PATypeHolder(arrty);
    return arrty;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoSetArrayToNull(llvm::Value* v)
{
    llvm::Value* len = DtoGEPi(v,0,0,"tmp",gIR->scopebb());
    llvm::Value* zerolen = llvm::ConstantInt::get(len->getType()->getContainedType(0), 0, false);
    new llvm::StoreInst(zerolen, len, gIR->scopebb());

    llvm::Value* ptr = DtoGEPi(v,0,1,"tmp",gIR->scopebb());
    const llvm::PointerType* pty = isaPointer(ptr->getType()->getContainedType(0));
    llvm::Value* nullptr = llvm::ConstantPointerNull::get(pty);
    new llvm::StoreInst(nullptr, ptr, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoArrayAssign(llvm::Value* dst, llvm::Value* src)
{
    assert(gIR);
    if (dst->getType() == src->getType())
    {
        llvm::Value* ptr = DtoGEPi(src,0,0,"tmp",gIR->scopebb());
        llvm::Value* val = new llvm::LoadInst(ptr,"tmp",gIR->scopebb());
        ptr = DtoGEPi(dst,0,0,"tmp",gIR->scopebb());
        new llvm::StoreInst(val, ptr, gIR->scopebb());

        ptr = DtoGEPi(src,0,1,"tmp",gIR->scopebb());
        val = new llvm::LoadInst(ptr,"tmp",gIR->scopebb());
        ptr = DtoGEPi(dst,0,1,"tmp",gIR->scopebb());
        new llvm::StoreInst(val, ptr, gIR->scopebb());
    }
    else
    {
        Logger::cout() << "array assignment type dont match: " << *dst->getType() << "\n\n" << *src->getType() << '\n';
        const llvm::ArrayType* arrty = isaArray(src->getType()->getContainedType(0));
        if (!arrty)
        {
            Logger::cout() << "invalid: " << *src << '\n';
            assert(0);
        }
        const llvm::Type* dstty = getPtrToType(arrty->getElementType());

        llvm::Value* dstlen = DtoGEPi(dst,0,0,"tmp",gIR->scopebb());
        llvm::Value* srclen = DtoConstSize_t(arrty->getNumElements());
        new llvm::StoreInst(srclen, dstlen, gIR->scopebb());

        llvm::Value* dstptr = DtoGEPi(dst,0,1,"tmp",gIR->scopebb());
        llvm::Value* srcptr = new llvm::BitCastInst(src,dstty,"tmp",gIR->scopebb());
        new llvm::StoreInst(srcptr, dstptr, gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoArrayInit(llvm::Value* l, llvm::Value* r)
{
    const llvm::PointerType* ptrty = isaPointer(l->getType());
    const llvm::Type* t = ptrty->getContainedType(0);
    const llvm::ArrayType* arrty = isaArray(t);
    if (arrty)
    {
        llvm::Value* ptr = DtoGEPi(l,0,0,"tmp",gIR->scopebb());
        llvm::Value* dim = llvm::ConstantInt::get(DtoSize_t(), arrty->getNumElements(), false);
        DtoArrayInit(ptr, dim, r);
    }
    else if (isaStruct(t))
    {
        llvm::Value* dim = DtoLoad(DtoGEPi(l, 0,0, "tmp"));
        llvm::Value* ptr = DtoLoad(DtoGEPi(l, 0,1, "tmp"));
        DtoArrayInit(ptr, dim, r);
    }
    else
    assert(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

typedef const llvm::Type* constLLVMTypeP;

static size_t checkRectArrayInit(const llvm::Type* pt, constLLVMTypeP& finalty)
{
    if (const llvm::ArrayType* arrty = isaArray(pt)) {
        size_t n = checkRectArrayInit(arrty->getElementType(), finalty);
        size_t ne = arrty->getNumElements();
        if (n) return n * ne;
        return ne;
    }
    finalty = pt;
    return 0;
}

void DtoArrayInit(llvm::Value* ptr, llvm::Value* dim, llvm::Value* val)
{
    Logger::cout() << "array: " << *ptr << " dim: " << *dim << " val: " << *val << '\n';
    const llvm::Type* pt = ptr->getType()->getContainedType(0);
    const llvm::Type* t = val->getType();
    const llvm::Type* finalTy;
    size_t aggrsz = 0;
    if (size_t arrsz = checkRectArrayInit(pt, finalTy)) {
        assert(finalTy == t);
        llvm::Constant* c = isaConstant(dim);
        assert(c);
        dim = llvm::ConstantExpr::getMul(c, DtoConstSize_t(arrsz));
        ptr = gIR->ir->CreateBitCast(ptr, getPtrToType(finalTy), "tmp");
    }
    else if (isaStruct(t)) {
        aggrsz = getABITypeSize(t);
        llvm::Constant* c = isaConstant(val);
        if (c && c->isNullValue()) {
            llvm::Value* nbytes;
            if (aggrsz == 1)
                nbytes = dim;
            else
                nbytes = gIR->ir->CreateMul(dim, DtoConstSize_t(aggrsz), "tmp");
            DtoMemSetZero(ptr,nbytes);
            return;
        }
        else {
            ptr = gIR->ir->CreateBitCast(ptr, getPtrToType(llvm::Type::Int8Ty), "tmp");
        }
    }
    else {
        assert(t == pt);
    }

    Logger::cout() << "array: " << *ptr << " dim: " << *dim << " val: " << *val << '\n';

    std::vector<llvm::Value*> args;
    args.push_back(ptr);
    args.push_back(dim);
    args.push_back(val);

    const char* funcname = NULL;

    if (aggrsz) {
        funcname = "_d_array_init_mem";
        args.push_back(DtoConstSize_t(aggrsz));
    }
    else if (isaPointer(t)) {
        funcname = "_d_array_init_pointer";

        const llvm::Type* dstty = getPtrToType(getPtrToType(llvm::Type::Int8Ty));
        if (args[0]->getType() != dstty)
            args[0] = new llvm::BitCastInst(args[0],dstty,"tmp",gIR->scopebb());

        const llvm::Type* valty = getPtrToType(llvm::Type::Int8Ty);
        if (args[2]->getType() != valty)
            args[2] = new llvm::BitCastInst(args[2],valty,"tmp",gIR->scopebb());
    }
    else if (t == llvm::Type::Int1Ty) {
        funcname = "_d_array_init_i1";
    }
    else if (t == llvm::Type::Int8Ty) {
        funcname = "_d_array_init_i8";
    }
    else if (t == llvm::Type::Int16Ty) {
        funcname = "_d_array_init_i16";
    }
    else if (t == llvm::Type::Int32Ty) {
        funcname = "_d_array_init_i32";
    }
    else if (t == llvm::Type::Int64Ty) {
        funcname = "_d_array_init_i64";
    }
    else if (t == llvm::Type::FloatTy) {
        funcname = "_d_array_init_float";
    }
    else if (t == llvm::Type::DoubleTy) {
        funcname = "_d_array_init_double";
    }
    else {
        Logger::cout() << *ptr->getType() << " = " << *val->getType() << '\n';
        assert(0);
    }

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, funcname);
    assert(fn);
    Logger::cout() << "calling array init function: " << *fn <<'\n';
    llvm::CallInst* call = new llvm::CallInst(fn, args.begin(), args.end(), "", gIR->scopebb());
    call->setCallingConv(llvm::CallingConv::C);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr)
{
    Logger::cout() << "DtoSetArray(" << *arr << ", " << *dim << ", " << *ptr << ")\n";
    const llvm::StructType* st = isaStruct(arr->getType()->getContainedType(0));

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* arrdim = DtoGEP(arr,zero,zero,"tmp",gIR->scopebb());
    new llvm::StoreInst(dim, arrdim, gIR->scopebb());

    llvm::Value* arrptr = DtoGEP(arr,zero,one,"tmp",gIR->scopebb());
    new llvm::StoreInst(ptr, arrptr, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* DtoConstArrayInitializer(ArrayInitializer* arrinit)
{
    Logger::println("DtoConstArrayInitializer: %s | %s", arrinit->toChars(), arrinit->type->toChars());
    LOG_SCOPE;

    Type* arrinittype = DtoDType(arrinit->type);

    Type* t;
    integer_t tdim;
    if (arrinittype->ty == Tsarray) {
        Logger::println("static array");
        TypeSArray* tsa = (TypeSArray*)arrinittype;
        tdim = tsa->dim->toInteger();
        t = tsa;
    }
    else if (arrinittype->ty == Tarray) {
        Logger::println("dynamic array");
        t = arrinittype;
        tdim = arrinit->dim;
    }
    else
    assert(0);

    Logger::println("dim = %u", tdim);

    std::vector<llvm::Constant*> inits(tdim, NULL);

    Type* arrnext = arrinittype->next;
    const llvm::Type* elemty = DtoType(arrinittype->next);

    assert(arrinit->index.dim == arrinit->value.dim);
    for (unsigned i=0,j=0; i < tdim; ++i)
    {
        Initializer* init = 0;
        Expression* idx;

        if (j < arrinit->index.dim)
            idx = (Expression*)arrinit->index.data[j];
        else
            idx = NULL;

        llvm::Constant* v = NULL;

        if (idx)
        {
            Logger::println("%d has idx", i);
            // this is pretty weird :/ idx->type turned out NULL for the initializer:
            //     const in6_addr IN6ADDR_ANY = { s6_addr8: [0] };
            // in std.c.linux.socket
            if (idx->type) {
                Logger::println("has idx->type", i);
                //integer_t k = idx->toInteger();
                //Logger::println("getting value for exp: %s | %s", idx->toChars(), arrnext->toChars());
                llvm::Constant* cc = idx->toConstElem(gIR);
                Logger::println("value gotten");
                assert(cc != NULL);
                llvm::ConstantInt* ci = llvm::dyn_cast<llvm::ConstantInt>(cc);
                assert(ci != NULL);
                uint64_t k = ci->getZExtValue();
                if (i == k)
                {
                    init = (Initializer*)arrinit->value.data[j];
                    assert(init);
                    ++j;
                }
            }
        }
        else
        {
            if (j < arrinit->value.dim) {
                init = (Initializer*)arrinit->value.data[j];
                ++j;
            }
            else
                v = arrnext->defaultInit()->toConstElem(gIR);
        }

        if (!v)
            v = DtoConstInitializer(t->next, init);
        assert(v);

        inits[i] = v;
    }

    const llvm::ArrayType* arrty = llvm::ArrayType::get(elemty,tdim);
    llvm::Constant* constarr = llvm::ConstantArray::get(arrty, inits);

    if (arrinittype->ty == Tsarray)
        return constarr;
    else
        assert(arrinittype->ty == Tarray);

    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(arrty,true,llvm::GlobalValue::InternalLinkage,constarr,"constarray",gIR->module);
    llvm::Constant* idxs[2] = { DtoConstUint(0), DtoConstUint(0) };
    llvm::Constant* gep = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);
    return DtoConstSlice(DtoConstSize_t(tdim),gep);
}

//////////////////////////////////////////////////////////////////////////////////////////
static llvm::Value* get_slice_ptr(DSliceValue* e, llvm::Value*& sz)
{
    const llvm::Type* t = e->ptr->getType()->getContainedType(0);
    llvm::Value* ret = 0;
    if (e->len != 0) {
        // this means it's a real slice
        ret = e->ptr;

        size_t elembsz = getABITypeSize(ret->getType());
        llvm::ConstantInt* elemsz = llvm::ConstantInt::get(DtoSize_t(), elembsz, false);

        if (isaConstantInt(e->len)) {
            sz = llvm::ConstantExpr::getMul(elemsz, isaConstant(e->len));
        }
        else {
            sz = llvm::BinaryOperator::createMul(elemsz,e->len,"tmp",gIR->scopebb());
        }
    }
    else if (isaArray(t)) {
        ret = DtoGEPi(e->ptr, 0, 0, "tmp", gIR->scopebb());

        size_t elembsz = getABITypeSize(ret->getType()->getContainedType(0));
        llvm::ConstantInt* elemsz = llvm::ConstantInt::get(DtoSize_t(), elembsz, false);

        size_t numelements = isaArray(t)->getNumElements();
        llvm::ConstantInt* nelems = llvm::ConstantInt::get(DtoSize_t(), numelements, false);

        sz = llvm::ConstantExpr::getMul(elemsz, nelems);
    }
    else if (isaStruct(t)) {
        ret = DtoGEPi(e->ptr, 0, 1, "tmp", gIR->scopebb());
        ret = new llvm::LoadInst(ret, "tmp", gIR->scopebb());

        size_t elembsz = getABITypeSize(ret->getType()->getContainedType(0));
        llvm::ConstantInt* elemsz = llvm::ConstantInt::get(DtoSize_t(), elembsz, false);

        llvm::Value* len = DtoGEPi(e->ptr, 0, 0, "tmp", gIR->scopebb());
        len = new llvm::LoadInst(len, "tmp", gIR->scopebb());
        sz = llvm::BinaryOperator::createMul(len,elemsz,"tmp",gIR->scopebb());
    }
    else {
        assert(0);
    }
    return ret;
}

void DtoArrayCopySlices(DSliceValue* dst, DSliceValue* src)
{
    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);

    llvm::Value* sz1;
    llvm::Value* dstarr = new llvm::BitCastInst(get_slice_ptr(dst,sz1),arrty,"tmp",gIR->scopebb());

    llvm::Value* sz2;
    llvm::Value* srcarr = new llvm::BitCastInst(get_slice_ptr(src,sz2),arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemCpy64() : LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = sz1;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

void DtoArrayCopyToSlice(DSliceValue* dst, DValue* src)
{
    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);

    llvm::Value* sz1;
    llvm::Value* dstarr = new llvm::BitCastInst(get_slice_ptr(dst,sz1),arrty,"tmp",gIR->scopebb());
    llvm::Value* srcarr = new llvm::BitCastInst(DtoArrayPtr(src),arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemCpy64() : LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = sz1;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoStaticArrayCopy(llvm::Value* dst, llvm::Value* src)
{
    Logger::cout() << "static array copy: " << *dst << " from " << *src << '\n';
    assert(dst->getType() == src->getType());
    size_t arrsz = getABITypeSize(dst->getType()->getContainedType(0));
    llvm::Value* n = llvm::ConstantInt::get(DtoSize_t(), arrsz, false);

    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);
    llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
    llvm::Value* srcarr = new llvm::BitCastInst(src,arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = (global.params.is64bit) ? LLVM_DeclareMemCpy64() : LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = n;
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* DtoConstSlice(llvm::Constant* dim, llvm::Constant* ptr)
{
    std::vector<const llvm::Type*> types;
    types.push_back(dim->getType());
    types.push_back(ptr->getType());
    const llvm::StructType* type = llvm::StructType::get(types);
    std::vector<llvm::Constant*> values;
    values.push_back(dim);
    values.push_back(ptr);
    return llvm::ConstantStruct::get(type,values);
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoNewDynArray(llvm::Value* dst, llvm::Value* dim, Type* dty, bool doinit)
{
    const llvm::Type* ty = DtoType(dty);
    assert(ty != llvm::Type::VoidTy);
    size_t sz = getABITypeSize(ty);
    llvm::ConstantInt* n = llvm::ConstantInt::get(DtoSize_t(), sz, false);
    llvm::Value* bytesize = (sz == 1) ? dim : llvm::BinaryOperator::createMul(n,dim,"tmp",gIR->scopebb());

    llvm::Value* nullptr = llvm::ConstantPointerNull::get(getPtrToType(ty));

    llvm::Value* newptr = DtoRealloc(nullptr, bytesize);

    if (doinit) {
        DValue* e = dty->defaultInit()->toElem(gIR);
        DtoArrayInit(newptr,dim,e->getRVal());
    }

    llvm::Value* lenptr = DtoGEPi(dst,0,0,"tmp",gIR->scopebb());
    new llvm::StoreInst(dim,lenptr,gIR->scopebb());
    llvm::Value* ptrptr = DtoGEPi(dst,0,1,"tmp",gIR->scopebb());
    new llvm::StoreInst(newptr,ptrptr,gIR->scopebb());

    return newptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoResizeDynArray(llvm::Value* arr, llvm::Value* sz)
{
    llvm::Value* ptr = DtoGEPi(arr, 0, 1, "tmp", gIR->scopebb());
    llvm::Value* ptrld = new llvm::LoadInst(ptr,"tmp",gIR->scopebb());

    size_t isz = getABITypeSize(ptrld->getType()->getContainedType(0));
    llvm::ConstantInt* n = llvm::ConstantInt::get(DtoSize_t(), isz, false);
    llvm::Value* bytesz = (isz == 1) ? sz : llvm::BinaryOperator::createMul(n,sz,"tmp",gIR->scopebb());

    llvm::Value* newptr = DtoRealloc(ptrld, bytesz);
    new llvm::StoreInst(newptr,ptr,gIR->scopebb());

    llvm::Value* len = DtoGEPi(arr, 0, 0, "tmp", gIR->scopebb());
    new llvm::StoreInst(sz,len,gIR->scopebb());

    return newptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoCatAssignElement(llvm::Value* arr, Expression* exp)
{
    llvm::Value* ptr = DtoGEPi(arr, 0, 0, "tmp");
    llvm::Value* idx = DtoLoad(ptr);
    llvm::Value* one = llvm::ConstantInt::get(idx->getType(),1,false);
    llvm::Value* len = llvm::BinaryOperator::createAdd(idx, one, "tmp", gIR->scopebb());
    DtoResizeDynArray(arr,len);

    ptr = DtoLoad(DtoGEPi(arr, 0, 1, "tmp"));
    ptr = new llvm::GetElementPtrInst(ptr, idx, "tmp", gIR->scopebb());

    DValue* dptr = new DVarValue(exp->type, ptr, true);

    gIR->exps.push_back(IRExp(0,exp,dptr));
    DValue* e = exp->toElem(gIR);
    gIR->exps.pop_back();

    if (!e->inPlace())
        DtoAssign(dptr, e);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoCatAssignArray(llvm::Value* arr, Expression* exp)
{
    DValue* e = exp->toElem(gIR);

    llvm::Value *len1, *len2, *src1, *src2, *res;

    DValue* darr = new DVarValue(exp->type, arr, true);

    len1 = DtoArrayLen(darr);
    len2 = DtoArrayLen(e);
    res = gIR->ir->CreateAdd(len1,len2,"tmp");

    llvm::Value* mem = DtoResizeDynArray(arr,res);

    src1 = DtoArrayPtr(darr);
    src2 = DtoArrayPtr(e);

    mem = gIR->ir->CreateGEP(mem,len1,"tmp");
    DtoMemCpy(mem,src2,len2);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoCatArrays(llvm::Value* arr, Expression* exp1, Expression* exp2)
{
    Type* t1 = DtoDType(exp1->type);
    Type* t2 = DtoDType(exp2->type);

    assert(t1->ty == Tarray);
    assert(t1->ty == t2->ty);

    DValue* e1 = exp1->toElem(gIR);
    llvm::Value* a = e1->getRVal();

    DValue* e2 = exp2->toElem(gIR);
    llvm::Value* b = e2->getRVal();

    llvm::Value *len1, *len2, *src1, *src2, *res;
    len1 = gIR->ir->CreateLoad(DtoGEPi(a,0,0,"tmp"),"tmp");
    len2 = gIR->ir->CreateLoad(DtoGEPi(b,0,0,"tmp"),"tmp");
    res = gIR->ir->CreateAdd(len1,len2,"tmp");

    llvm::Value* mem = DtoNewDynArray(arr, res, DtoDType(t1->next), false);

    src1 = gIR->ir->CreateLoad(DtoGEPi(a,0,1,"tmp"),"tmp");
    src2 = gIR->ir->CreateLoad(DtoGEPi(b,0,1,"tmp"),"tmp");

    DtoMemCpy(mem,src1,len1);
    mem = gIR->ir->CreateGEP(mem,len1,"tmp");
    DtoMemCpy(mem,src2,len2);
}

//////////////////////////////////////////////////////////////////////////////////////////
void DtoCatArrayElement(llvm::Value* arr, Expression* exp1, Expression* exp2)
{
    Type* t1 = DtoDType(exp1->type);
    Type* t2 = DtoDType(exp2->type);

    assert(t1->ty == Tarray);
    assert(t2 == DtoDType(t1->next));

    DValue* e1 = exp1->toElem(gIR);
    DValue* e2 = exp2->toElem(gIR);

    llvm::Value *len1, *src1, *res;
    llvm::Value* a = e1->getRVal();
    len1 = gIR->ir->CreateLoad(DtoGEPi(a,0,0,"tmp"),"tmp");
    res = gIR->ir->CreateAdd(len1,DtoConstSize_t(1),"tmp");

    llvm::Value* mem = DtoNewDynArray(arr, res, DtoDType(t1->next), false);

    src1 = gIR->ir->CreateLoad(DtoGEPi(a,0,1,"tmp"),"tmp");

    DtoMemCpy(mem,src1,len1);

    mem = gIR->ir->CreateGEP(mem,len1,"tmp");
    DVarValue* memval = new DVarValue(e2->getType(), mem, true);
    DtoAssign(memval, e2);
}

//////////////////////////////////////////////////////////////////////////////////////////
// helper for eq and cmp
static llvm::Value* DtoArrayEqCmp_impl(const char* func, DValue* l, DValue* r, bool useti)
{
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, func);
    assert(fn);

    llvm::Value* lmem;
    llvm::Value* rmem;

    // cast static arrays to dynamic ones, this turns them into DSliceValues
    Type* l_ty = DtoDType(l->getType());
    Type* r_ty = DtoDType(r->getType());
    assert(l_ty->next == r_ty->next);
    if ((l_ty->ty == Tsarray) || (r_ty->ty == Tsarray)) {
        Type* a_ty = new Type(Tarray, l_ty->next);
        if (l_ty->ty == Tsarray)
            l = DtoCastArray(l, a_ty);
        if (r_ty->ty == Tsarray)
            r = DtoCastArray(r, a_ty);
    }

    // we need to give slices storage
    if (l->isSlice()) {
        lmem = new llvm::AllocaInst(DtoType(l->getType()), "tmpparam", gIR->topallocapoint());
        DtoSetArray(lmem, DtoArrayLen(l), DtoArrayPtr(l));
    }
    else
        lmem = l->getRVal();

    if (r->isSlice()) {
        rmem = new llvm::AllocaInst(DtoType(r->getType()), "tmpparam", gIR->topallocapoint());
        DtoSetArray(rmem, DtoArrayLen(r), DtoArrayPtr(r));
    }
    else
        rmem = r->getRVal();

    const llvm::Type* pt = fn->getFunctionType()->getParamType(0);

    std::vector<llvm::Value*> args;
    args.push_back(DtoBitCast(lmem,pt));
    args.push_back(DtoBitCast(rmem,pt));

    // pass element typeinfo ?
    if (useti) {
        TypeInfoDeclaration* ti = DtoDType(l->getType())->next->getTypeInfoDeclaration();
        DtoForceConstInitDsymbol(ti);
        Logger::cout() << "typeinfo decl: " << *gIR->irDsymbol[ti].getIrValue() << '\n';

        pt = fn->getFunctionType()->getParamType(2);
        args.push_back(DtoBitCast(gIR->irDsymbol[ti].getIrValue(), pt));
    }

    return gIR->ir->CreateCall(fn, args.begin(), args.end(), "tmp");
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoArrayEquals(TOK op, DValue* l, DValue* r)
{
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_adEq");
    assert(fn);

    llvm::Value* res = DtoArrayEqCmp_impl("_adEq", l, r, true);
    if (op == TOKnotequal)
        res = gIR->ir->CreateNot(res, "tmp");

    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoArrayCompare(TOK op, DValue* l, DValue* r)
{
    llvm::Value* res = 0;

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
        res = llvm::ConstantInt::getTrue();
        break;
    case TOKunord:
        skip = true;
        res = llvm::ConstantInt::getFalse();
        break;

    default:
        assert(0);
    }

    if (!skip)
    {
        Type* t = DtoDType(DtoDType(l->getType())->next);
        if (t->ty == Tchar)
            res = DtoArrayEqCmp_impl("_adCmpChar", l, r, false);
        else
            res = DtoArrayEqCmp_impl("_adCmp", l, r, true);
        res = new llvm::ICmpInst(cmpop, res, DtoConstInt(0), "tmp", gIR->scopebb());
    }

    assert(res);
    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoArrayCastLength(llvm::Value* len, const llvm::Type* elemty, const llvm::Type* newelemty)
{
    size_t esz = getABITypeSize(elemty);
    size_t nsz = getABITypeSize(newelemty);
    if (esz == nsz)
        return len;

    std::vector<llvm::Value*> args;
    args.push_back(len);
    args.push_back(llvm::ConstantInt::get(DtoSize_t(), esz, false));
    args.push_back(llvm::ConstantInt::get(DtoSize_t(), nsz, false));

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_array_cast_len");
    return new llvm::CallInst(fn, args.begin(), args.end(), "tmp", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoDynArrayIs(TOK op, llvm::Value* l, llvm::Value* r)
{
    llvm::ICmpInst::Predicate pred = (op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;

    if (r == NULL) {
        llvm::Value* ll = gIR->ir->CreateLoad(DtoGEPi(l, 0,0, "tmp"),"tmp");
        llvm::Value* rl = DtoConstSize_t(0);
        llvm::Value* b1 = gIR->ir->CreateICmp(pred,ll,rl,"tmp");

        llvm::Value* lp = gIR->ir->CreateLoad(DtoGEPi(l, 0,1, "tmp"),"tmp");
        const llvm::PointerType* pty = isaPointer(lp->getType());
        llvm::Value* rp = llvm::ConstantPointerNull::get(pty);
        llvm::Value* b2 = gIR->ir->CreateICmp(pred,lp,rp,"tmp");

        llvm::Value* b = gIR->ir->CreateAnd(b1,b2,"tmp");
        return b;
    }
    else {
        assert(l->getType() == r->getType());

        llvm::Value* ll = gIR->ir->CreateLoad(DtoGEPi(l, 0,0, "tmp"),"tmp");
        llvm::Value* rl = gIR->ir->CreateLoad(DtoGEPi(r, 0,0, "tmp"),"tmp");
        llvm::Value* b1 = gIR->ir->CreateICmp(pred,ll,rl,"tmp");

        llvm::Value* lp = gIR->ir->CreateLoad(DtoGEPi(l, 0,1, "tmp"),"tmp");
        llvm::Value* rp = gIR->ir->CreateLoad(DtoGEPi(r, 0,1, "tmp"),"tmp");
        llvm::Value* b2 = gIR->ir->CreateICmp(pred,lp,rp,"tmp");

        llvm::Value* b = gIR->ir->CreateAnd(b1,b2,"tmp");
        return b;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* DtoConstStaticArray(const llvm::Type* t, llvm::Constant* c)
{
    const llvm::ArrayType* at = isaArray(t);
    assert(at);

    if (isaArray(at->getElementType()))
    {
        c = DtoConstStaticArray(at->getElementType(), c);
    }
    else {
        assert(at->getElementType() == c->getType());
    }
    std::vector<llvm::Constant*> initvals;
    initvals.resize(at->getNumElements(), c);
    return llvm::ConstantArray::get(at, initvals);
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoArrayLen(DValue* v)
{
    Logger::println("DtoArrayLen");
    LOG_SCOPE;
    Type* t = DtoDType(v->getType());
    if (t->ty == Tarray) {
        if (DSliceValue* s = v->isSlice()) {
            if (s->len) {
                return s->len;
            }
            const llvm::ArrayType* arrTy = isaArray(s->ptr->getType()->getContainedType(0));
            if (arrTy)
                return DtoConstSize_t(arrTy->getNumElements());
            else
                return DtoLoad(DtoGEPi(s->ptr, 0,0, "tmp"));
        }
        return DtoLoad(DtoGEPi(v->getRVal(), 0,0, "tmp"));
    }
    else if (t->ty == Tsarray) {
        assert(!v->isSlice());
        llvm::Value* rv = v->getRVal();
        Logger::cout() << "casting: " << *rv << '\n';
        const llvm::ArrayType* t = isaArray(rv->getType()->getContainedType(0));
        return DtoConstSize_t(t->getNumElements());
    }
    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Value* DtoArrayPtr(DValue* v)
{
    Type* t = DtoDType(v->getType());
    if (t->ty == Tarray) {
        if (DSliceValue* s = v->isSlice()) {
            if (s->len) return s->ptr;
            const llvm::Type* t = s->ptr->getType()->getContainedType(0);
            Logger::cout() << "ptr of full slice: " << *s->ptr << '\n';
            const llvm::ArrayType* arrTy = isaArray(s->ptr->getType()->getContainedType(0));
            if (arrTy)
                return DtoGEPi(s->ptr, 0,0, "tmp");
            else
                return DtoLoad(DtoGEPi(s->ptr, 0,1, "tmp"));
        }
        return DtoLoad(DtoGEPi(v->getRVal(), 0,1, "tmp"));
    }
    else if (t->ty == Tsarray) {
        return DtoGEPi(v->getRVal(), 0,0, "tmp");
    }
    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
DValue* DtoCastArray(DValue* u, Type* to)
{
    const llvm::Type* tolltype = DtoType(to);

    Type* totype = DtoDType(to);
    Type* fromtype = DtoDType(u->getType());
    assert(fromtype->ty == Tarray || fromtype->ty == Tsarray);

    llvm::Value* rval;
    llvm::Value* rval2;
    bool isslice = false;

    Logger::cout() << "from array or sarray" << '\n';
    if (totype->ty == Tpointer) {
        Logger::cout() << "to pointer" << '\n';
        assert(fromtype->next == totype->next || totype->next->ty == Tvoid);
        llvm::Value* ptr = DtoGEPi(u->getRVal(),0,1,"tmp",gIR->scopebb());
        rval = new llvm::LoadInst(ptr, "tmp", gIR->scopebb());
        if (fromtype->next != totype->next)
            rval = gIR->ir->CreateBitCast(rval, getPtrToType(llvm::Type::Int8Ty), "tmp");
    }
    else if (totype->ty == Tarray) {
        Logger::cout() << "to array" << '\n';
        const llvm::Type* ptrty = DtoType(totype->next);
        if (ptrty == llvm::Type::VoidTy)
            ptrty = llvm::Type::Int8Ty;
        ptrty = getPtrToType(ptrty);

        const llvm::Type* ety = DtoType(fromtype->next);
        if (ety == llvm::Type::VoidTy)
            ety = llvm::Type::Int8Ty;

        if (DSliceValue* usl = u->isSlice()) {
            Logger::println("from slice");
            Logger::cout() << "from: " << *usl->ptr << " to: " << *ptrty << '\n';
            rval = new llvm::BitCastInst(usl->ptr, ptrty, "tmp", gIR->scopebb());
            if (fromtype->next->size() == totype->next->size())
                rval2 = usl->len;
            else
                rval2 = DtoArrayCastLength(usl->len, ety, ptrty->getContainedType(0));
        }
        else {
            llvm::Value* uval = u->getRVal();
            if (fromtype->ty == Tsarray) {
                Logger::cout() << "uvalTy = " << *uval->getType() << '\n';
                assert(isaPointer(uval->getType()));
                const llvm::ArrayType* arrty = isaArray(uval->getType()->getContainedType(0));
                rval2 = llvm::ConstantInt::get(DtoSize_t(), arrty->getNumElements(), false);
                rval2 = DtoArrayCastLength(rval2, ety, ptrty->getContainedType(0));
                rval = new llvm::BitCastInst(uval, ptrty, "tmp", gIR->scopebb());
            }
            else {
                llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
                llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
                rval2 = DtoGEP(uval,zero,zero,"tmp",gIR->scopebb());
                rval2 = new llvm::LoadInst(rval2, "tmp", gIR->scopebb());
                rval2 = DtoArrayCastLength(rval2, ety, ptrty->getContainedType(0));

                rval = DtoGEP(uval,zero,one,"tmp",gIR->scopebb());
                rval = new llvm::LoadInst(rval, "tmp", gIR->scopebb());
                //Logger::cout() << *e->mem->getType() << '|' << *ptrty << '\n';
                rval = new llvm::BitCastInst(rval, ptrty, "tmp", gIR->scopebb());
            }
        }
        isslice = true;
    }
    else if (totype->ty == Tsarray) {
        Logger::cout() << "to sarray" << '\n';
        assert(0);
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
