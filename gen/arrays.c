#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/CallingConv.h"

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
#include "gen/elem.h"

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* LLVM_DtoArrayType(Type* t)
{
    assert(t->next);
    const llvm::Type* at = LLVM_DtoType(t->next);
    const llvm::Type* arrty;

    /*if (t->ty == Tsarray) {
        TypeSArray* tsa = (TypeSArray*)t;
        assert(tsa->dim->type->isintegral());
        arrty = llvm::ArrayType::get(at,tsa->dim->toUInteger());
    }
    else {
        arrty = llvm::ArrayType::get(at,0);
    }*/
    if (at == llvm::Type::VoidTy) {
        at = llvm::Type::Int8Ty;
    }
    arrty = llvm::PointerType::get(at);

    std::vector<const llvm::Type*> members;
    if (global.params.is64bit)
        members.push_back(llvm::Type::Int64Ty);
    else
        members.push_back(llvm::Type::Int32Ty);

    members.push_back(arrty);

    return llvm::StructType::get(members);
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::ArrayType* LLVM_DtoStaticArrayType(Type* t)
{
    if (t->llvmType)
        return llvm::cast<llvm::ArrayType>(t->llvmType);

    assert(t->ty == Tsarray);
    assert(t->next);

    const llvm::Type* at = LLVM_DtoType(t->next);

    TypeSArray* tsa = (TypeSArray*)t;
    assert(tsa->dim->type->isintegral());
    const llvm::ArrayType* arrty = llvm::ArrayType::get(at,tsa->dim->toUInteger());

    tsa->llvmType = arrty;
    return arrty;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoNullArray(llvm::Value* v)
{
    assert(gIR);
    d_uns64 n = (global.params.is64bit) ? 16 : 8;

    llvm::Type* i8p_ty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* arr = new llvm::BitCastInst(v,i8p_ty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemSet32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = arr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    //Logger::cout() << *fn << '|' << *fn->getType() << '\n';
    //Logger::cout() << "to null array call: " << *llargs[0] << '|' << *llargs[1] << '|' << *llargs[2] << '|' << *llargs[3] << '\n';

    llvm::Value* ret = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoArrayAssign(llvm::Value* dst, llvm::Value* src)
{
    assert(gIR);
    if (dst->getType() == src->getType())
    {
        d_uns64 n = (global.params.is64bit) ? 16 : 8;

        llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

        llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
        llvm::Value* srcarr = new llvm::BitCastInst(src,arrty,"tmp",gIR->scopebb());

        llvm::Function* fn = LLVM_DeclareMemCpy32();
        std::vector<llvm::Value*> llargs;
        llargs.resize(4);
        llargs[0] = dstarr;
        llargs[1] = srcarr;
        llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
        llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

        return new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
    }
    else
    {
        if (!llvm::isa<llvm::ArrayType>(src->getType()->getContainedType(0)))
        {
            Logger::cout() << "invalid: " << *src << '\n';
            assert(0);
        }
        const llvm::ArrayType* arrty = llvm::cast<llvm::ArrayType>(src->getType()->getContainedType(0));
        llvm::Type* dstty = llvm::PointerType::get(arrty->getElementType());

        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
        llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

        llvm::Value* dstlen = LLVM_DtoGEP(dst,zero,zero,"tmp",gIR->scopebb());
        llvm::Value* srclen = llvm::ConstantInt::get(LLVM_DtoSize_t(), arrty->getNumElements(), false);
        new llvm::StoreInst(srclen, dstlen, gIR->scopebb());

        llvm::Value* dstptr = LLVM_DtoGEP(dst,zero,one,"tmp",gIR->scopebb());
        llvm::Value* srcptr = new llvm::BitCastInst(src,dstty,"tmp",gIR->scopebb());
        new llvm::StoreInst(srcptr, dstptr, gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoArrayInit(llvm::Value* l, llvm::Value* r)
{
    const llvm::PointerType* ptrty = llvm::cast<llvm::PointerType>(l->getType());
    if (llvm::isa<llvm::ArrayType>(ptrty->getContainedType(0)))
    {
        const llvm::ArrayType* arrty = llvm::cast<llvm::ArrayType>(ptrty->getContainedType(0));
        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

        std::vector<llvm::Value*> args;
        args.resize(3);
        args[0] = LLVM_DtoGEP(l,zero,zero,"tmp",gIR->scopebb());
        args[1] = llvm::ConstantInt::get(LLVM_DtoSize_t(), arrty->getNumElements(), false);
        args[2] = r;
        
        const char* funcname = NULL;
        
        if (llvm::isa<llvm::PointerType>(arrty->getElementType())) {
            funcname = "_d_array_init_pointer";
            
            const llvm::Type* dstty = llvm::PointerType::get(llvm::PointerType::get(llvm::Type::Int8Ty));
            if (args[0]->getType() != dstty)
                args[0] = new llvm::BitCastInst(args[0],dstty,"tmp",gIR->scopebb());
            
            const llvm::Type* valty = llvm::PointerType::get(llvm::Type::Int8Ty);
            if (args[2]->getType() != valty)
                args[2] = new llvm::BitCastInst(args[2],valty,"tmp",gIR->scopebb());
        }
        else if (r->getType() == llvm::Type::Int1Ty) {
            funcname = "_d_array_init_i1";
        }
        else if (r->getType() == llvm::Type::Int8Ty) {
            funcname = "_d_array_init_i8";
        }
        else if (r->getType() == llvm::Type::Int16Ty) {
            funcname = "_d_array_init_i16";
        }
        else if (r->getType() == llvm::Type::Int32Ty) {
            funcname = "_d_array_init_i32";
        }
        else if (r->getType() == llvm::Type::Int64Ty) {
            funcname = "_d_array_init_i64";
        }
        else if (r->getType() == llvm::Type::FloatTy) {
            funcname = "_d_array_init_float";
        }
        else if (r->getType() == llvm::Type::DoubleTy) {
            funcname = "_d_array_init_double";
        }
        else {
            assert(0);
        }
        
        Logger::cout() << *args[0] << '|' << *args[2] << '\n';
        
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, funcname);
        assert(fn);
        llvm::CallInst* call = new llvm::CallInst(fn, args.begin(), args.end(), "", gIR->scopebb());
        call->setCallingConv(llvm::CallingConv::C);
        
        Logger::println("array init call ok");
    }
    else if (llvm::isa<llvm::StructType>(ptrty->getContainedType(0)))
    {
        assert(0 && "Only static arrays support initialisers atm");
    }
    else
    assert(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoSetArray(llvm::Value* arr, llvm::Value* dim, llvm::Value* ptr)
{
    Logger::cout() << "LLVM_DtoSetArray(" << *arr << ", " << *dim << ", " << *ptr << ")\n";
    const llvm::StructType* st = llvm::cast<llvm::StructType>(arr->getType()->getContainedType(0));
    //const llvm::PointerType* pt = llvm::cast<llvm::PointerType>(r->getType());
    
    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* arrdim = LLVM_DtoGEP(arr,zero,zero,"tmp",gIR->scopebb());
    new llvm::StoreInst(dim, arrdim, gIR->scopebb());
    
    llvm::Value* arrptr = LLVM_DtoGEP(arr,zero,one,"tmp",gIR->scopebb());
    new llvm::StoreInst(ptr, arrptr, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* LLVM_DtoArrayInitializer(ArrayInitializer* arrinit)
{
    Logger::println("arr init begin");
    assert(arrinit->type->ty == Tsarray);
    TypeSArray* t = (TypeSArray*)arrinit->type;
    integer_t tdim = t->dim->toInteger();

    std::vector<llvm::Constant*> inits(tdim, 0);

    const llvm::Type* elemty = LLVM_DtoType(arrinit->type->next);

    assert(arrinit->index.dim == arrinit->value.dim);
    for (int i=0,j=0; i < tdim; ++i)
    {
        Initializer* init = 0;
        Expression* idx = (Expression*)arrinit->index.data[j];

        if (idx)
        {
            integer_t k = idx->toInteger();
            if (i == k)
            {
                init = (Initializer*)arrinit->value.data[j];
                assert(init);
                ++j;
            }
        }
        else
        {
            init = (Initializer*)arrinit->value.data[j];
            ++j;
        }

        llvm::Constant* v = 0;

        if (!init)
        {
            elem* e = t->next->defaultInit()->toElem(gIR);
            v = llvm::cast<llvm::Constant>(e->val);
            delete e;
        }
        else if (ExpInitializer* ex = init->isExpInitializer())
        {
            elem* e = ex->exp->toElem(gIR);
            v = llvm::cast<llvm::Constant>(e->val);
            delete e;
        }
        else if (StructInitializer* si = init->isStructInitializer())
        {
            v = LLVM_DtoStructInitializer(si);
        }
        else if (ArrayInitializer* ai = init->isArrayInitializer())
        {
            v = LLVM_DtoArrayInitializer(ai);
        }
        else if (init->isVoidInitializer())
        {
            v = llvm::UndefValue::get(elemty);
        }
        else
        assert(v);

        inits[i] = v;
    }

    const llvm::ArrayType* arrty = LLVM_DtoStaticArrayType(t);
    return llvm::ConstantArray::get(arrty, inits);
}

//////////////////////////////////////////////////////////////////////////////////////////
static llvm::Value* get_slice_ptr(elem* e, llvm::Value*& sz)
{
    assert(e->mem);
    const llvm::Type* t = e->mem->getType()->getContainedType(0);
    llvm::Value* ret = 0;
    if (llvm::isa<llvm::ArrayType>(t)) {
        ret = LLVM_DtoGEPi(e->mem, 0, 0, "tmp", gIR->scopebb());

        size_t elembsz = gTargetData->getTypeSize(ret->getType()->getContainedType(0));
        llvm::ConstantInt* elemsz = llvm::ConstantInt::get(LLVM_DtoSize_t(), elembsz, false);

        size_t numelements = llvm::cast<llvm::ArrayType>(t)->getNumElements();
        llvm::ConstantInt* nelems = llvm::ConstantInt::get(LLVM_DtoSize_t(), numelements, false);

        sz = llvm::ConstantExpr::getMul(elemsz, nelems);
    }
    else if (llvm::isa<llvm::StructType>(t)) {
        ret = LLVM_DtoGEPi(e->mem, 0, 1, "tmp", gIR->scopebb());
        ret = new llvm::LoadInst(ret, "tmp", gIR->scopebb());

        size_t elembsz = gTargetData->getTypeSize(ret->getType()->getContainedType(0));
        llvm::ConstantInt* elemsz = llvm::ConstantInt::get(LLVM_DtoSize_t(), elembsz, false);

        llvm::Value* len = LLVM_DtoGEPi(e->mem, 0, 0, "tmp", gIR->scopebb());
        len = new llvm::LoadInst(len, "tmp", gIR->scopebb());
        sz = llvm::BinaryOperator::createMul(len,elemsz,"tmp",gIR->scopebb());
    }
    else {
        assert(0);
    }
    return ret;
}

void LLVM_DtoArrayCopy(elem* dst, elem* src)
{
    Logger::cout() << "Array copy ((((" << *src->mem << ")))) into ((((" << *dst->mem << "))))\n";

    assert(dst->type == elem::SLICE);
    assert(src->type == elem::SLICE);

    llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* sz1;
    llvm::Value* sz2;
    llvm::Value* dstarr = new llvm::BitCastInst(get_slice_ptr(dst,sz1),arrty,"tmp",gIR->scopebb());
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

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* LLVM_DtoConstantSlice(llvm::Constant* dim, llvm::Constant* ptr)
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
