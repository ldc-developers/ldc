#include <iostream>

#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/CallingConv.h"

#include "mtype.h"
#include "dsymbol.h"
#include "aggregate.h"
#include "declaration.h"
#include "init.h"

#include "tollvm.h"
#include "irstate.h"
#include "logger.h"
#include "runtime.h"
#include "elem.h"

const llvm::Type* LLVM_DtoType(Type* t)
{
    assert(t);
    switch (t->ty)
    {
    // integers
    case Tint8:
    case Tuns8:
    case Tchar:
        return (const llvm::Type*)llvm::Type::Int8Ty;
    case Tint16:
    case Tuns16:
    case Twchar:
        return (const llvm::Type*)llvm::Type::Int16Ty;
    case Tint32:
    case Tuns32:
    case Tdchar:
        return (const llvm::Type*)llvm::Type::Int32Ty;
    case Tint64:
    case Tuns64:
        return (const llvm::Type*)llvm::Type::Int64Ty;

    case Tbool:
        return (const llvm::Type*)llvm::ConstantInt::getTrue()->getType();

    // floats
    case Tfloat32:
        return llvm::Type::FloatTy;
    case Tfloat64:
    case Tfloat80:
        return llvm::Type::DoubleTy;

    // pointers
    case Tpointer: {
        assert(t->next);
        if (t->next->ty == Tvoid)
            return (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
        else
            return (const llvm::Type*)llvm::PointerType::get(LLVM_DtoType(t->next));
    }

    // arrays
    case Tarray:
        return LLVM_DtoArrayType(t);
    case Tsarray:
        return LLVM_DtoStaticArrayType(t);

    // void
    case Tvoid:
        return llvm::Type::VoidTy;

    // aggregates
    case Tstruct:    {
        if (t->llvmType == 0)
        {
            // recursive or cyclic declaration
            if (!gIR->structs.empty())
            {
                IRStruct* found = 0;
                for (IRState::StructVector::iterator i=gIR->structs.begin(); i!=gIR->structs.end(); ++i)
                {
                    if (t == i->type)
                    {
                        return i->recty.get();
                    }
                }
            }

            // forward declaration
            TypeStruct* ts = (TypeStruct*)t;
            assert(ts->sym);
            ts->sym->toObjFile();
        }
        return t->llvmType;
    }

    case Tclass:    {
        if (t->llvmType == 0)
        {
            TypeClass* tc = (TypeClass*)t;
            assert(tc->sym);
            if (!tc->sym->llvmInProgress) {
                tc->sym->toObjFile();
            }
            else {
                //assert(0 && "circular class referencing");
                return llvm::OpaqueType::get();
            }
        }
        return llvm::PointerType::get(t->llvmType);
    }

    // functions
    case Tfunction:
    {
        if (t->llvmType == 0) {
            return LLVM_DtoFunctionType(t);
        }
        else {
            return t->llvmType;
        }
    }
    
    // delegates
    case Tdelegate:
    {
        if (t->llvmType == 0) {
            return LLVM_DtoDelegateType(t);
        }
        else {
            return t->llvmType;
        }
    }

    // typedefs
    case Ttypedef:
    {
        Type* bt = t->toBasetype();
        assert(bt);
        return LLVM_DtoType(bt);
    }

    default:
        printf("trying to convert unknown type with value %d\n", t->ty);
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::FunctionType* LLVM_DtoFunctionType(Type* t, const llvm::Type* thisparam)
{
    TypeFunction* f = (TypeFunction*)t;

    // parameter types
    const llvm::Type* rettype;
    std::vector<const llvm::Type*> paramvec;

    TY retty = f->next->ty;

    if (retty == Tstruct || retty == Tdelegate || retty == Tarray) {
        rettype = llvm::PointerType::get(LLVM_DtoType(f->next));
        paramvec.push_back(rettype);
        rettype = llvm::Type::VoidTy;
    }
    else {
        Type* rt = f->next;
        if (rt)
        rettype = LLVM_DtoType(rt);
        else
        assert(0);
    }

    if (thisparam) {
        paramvec.push_back(thisparam);
    }

    size_t n = Argument::dim(f->parameters);
    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = arg->type;
        assert(argT);
        paramvec.push_back(LLVM_DtoType(argT));
    }

    Logger::cout() << "Return type: " << *rettype << '\n';

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, paramvec, f->varargs);
    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::FunctionType* LLVM_DtoFunctionType(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)fdecl->type;
    assert(f != 0);

    // has already been pulled in by a reference to (
    if (f->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(f->llvmType);
    }

    // return value type
    const llvm::Type* rettype;
    const llvm::Type* actualRettype;
    Type* rt = f->next;
    bool retinptr = false;
    bool usesthis = false;

    if (fdecl->isMain()) {
        rettype = llvm::Type::Int32Ty;
        actualRettype = rettype;
    }
    else if (rt) {
        if (rt->ty == Tstruct || rt->ty == Tdelegate || rt->ty == Tarray) {
            rettype = llvm::PointerType::get(LLVM_DtoType(rt));
            actualRettype = llvm::Type::VoidTy;
            f->llvmRetInPtr = retinptr = true;
        }
        else {
            rettype = LLVM_DtoType(rt);
            actualRettype = rettype;
        }
    }
    else {
        assert(0);
    }

    // parameter types
    std::vector<const llvm::Type*> paramvec;

    if (retinptr) {
        Logger::print("returning through pointer parameter\n");
        paramvec.push_back(rettype);
    }

    if (fdecl->needThis() && fdecl->vthis) {
        Logger::print("this is: %s\n", fdecl->vthis->type->toChars());
        paramvec.push_back(LLVM_DtoType(fdecl->vthis->type));
        usesthis = true;
    }

    size_t n = Argument::dim(f->parameters);
    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = arg->type;
        assert(argT);

        if ((arg->storageClass & STCref) || (arg->storageClass & STCout)) {
            //assert(arg->vardecl);
            //arg->vardecl->refparam = true;
        }
        else
            arg->llvmCopy = true;

        const llvm::Type* at = LLVM_DtoType(argT);
        if (llvm::isa<llvm::StructType>(at)) {
            Logger::println("struct param");
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (llvm::isa<llvm::ArrayType>(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(llvm::PointerType::get(at->getContainedType(0)));
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else {
            if (!arg->llvmCopy) {
                Logger::println("ref param");
                at = llvm::PointerType::get(at);
            }
            else {
                Logger::println("in param");
            }
            paramvec.push_back(at);
        }
    }

    // construct function type
    bool isvararg = f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);

    f->llvmType = functype;
    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::StructType* LLVM_DtoDelegateType(Type* t)
{
    const llvm::Type* i8ptr = llvm::PointerType::get(llvm::Type::Int8Ty);
    const llvm::Type* func = LLVM_DtoFunctionType(t->next, i8ptr);
    const llvm::Type* funcptr = llvm::PointerType::get(func);

    std::vector<const llvm::Type*> types;
    types.push_back(i8ptr);
    types.push_back(funcptr);
    return llvm::StructType::get(types);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Type* LLVM_DtoStructType(Type* t)
{
    assert(0);
    std::vector<const llvm::Type*> types;
    return llvm::StructType::get(types);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::StructType* LLVM_DtoArrayType(Type* t)
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

llvm::ArrayType* LLVM_DtoStaticArrayType(Type* t)
{
    if (t->llvmType)
        return llvm::cast<llvm::ArrayType>(t->llvmType);

    assert(t->ty == Tsarray);
    assert(t->next);

    const llvm::Type* at = LLVM_DtoType(t->next);

    TypeSArray* tsa = (TypeSArray*)t;
    assert(tsa->dim->type->isintegral());
    llvm::ArrayType* arrty = llvm::ArrayType::get(at,tsa->dim->toUInteger());

    tsa->llvmType = arrty;
    return arrty;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* LLVM_DeclareMemIntrinsic(const char* name, int bits, bool set=false)
{
    assert(bits == 32 || bits == 64);
    const llvm::Type* int8ty =    (const llvm::Type*)llvm::Type::Int8Ty;
    const llvm::Type* int32ty =   (const llvm::Type*)llvm::Type::Int32Ty;
    const llvm::Type* int64ty =   (const llvm::Type*)llvm::Type::Int64Ty;
    const llvm::Type* int8ptrty = (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
    const llvm::Type* voidty =    (const llvm::Type*)llvm::Type::VoidTy;

    assert(gIR);
    assert(gIR->module);

    // parameter types
    std::vector<const llvm::Type*> pvec;
    pvec.push_back(int8ptrty);
    pvec.push_back(set?int8ty:int8ptrty);
    pvec.push_back(bits==32?int32ty:int64ty);
    pvec.push_back(int32ty);
    llvm::FunctionType* functype = llvm::FunctionType::get(voidty, pvec, false);
    return new llvm::Function(functype, llvm::GlobalValue::ExternalLinkage, name, gIR->module);
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memset.i32
static llvm::Function* LLVM_DeclareMemSet32()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memset.i32", 32, true);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* LLVM_DeclareMemSet64()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memset.i64", 64, true);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memcpy.i32
static llvm::Function* LLVM_DeclareMemCpy32()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memcpy.i32", 32);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

// llvm.memcpy.i64
static llvm::Function* LLVM_DeclareMemCpy64()
{
    static llvm::Function* _func = 0;
    if (_func == 0) {
        _func = LLVM_DeclareMemIntrinsic("llvm.memcpy.i64", 64);
    }
    return _func;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoStructZeroInit(TypeStruct* t, llvm::Value* v)
{
    assert(gIR);
    uint64_t n = gTargetData->getTypeSize(t->llvmType);
    //llvm::Type* sarrty = llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::Int8Ty, n));
    llvm::Type* sarrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* sarr = new llvm::BitCastInst(v,sarrty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemSet32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = sarr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    llvm::Value* ret = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoStructCopy(TypeStruct* t, llvm::Value* dst, llvm::Value* src)
{
    assert(dst->getType() == src->getType());
    assert(gIR);

    uint64_t n = gTargetData->getTypeSize(t->llvmType);
    //llvm::Type* sarrty = llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::Int8Ty, n));
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

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* LLVM_DtoStructInitializer(StructInitializer* si)
{
    llvm::StructType* structtype = llvm::cast<llvm::StructType>(si->ad->llvmType);
    size_t n = structtype->getNumElements();

    assert(si->value.dim == si->vars.dim);

    std::vector<llvm::Constant*> inits;
    inits.resize(n, NULL);
    for (int i = 0; i < si->value.dim; ++i)
    {
        Initializer* ini = (Initializer*)si->value.data[i];
        assert(ini);

        VarDeclaration* vd = (VarDeclaration*)si->vars.data[i];
        assert(vd);
        Logger::println("vars[%d] = %s", i, vd->toChars());
        unsigned idx = si->ad->offsetToIndex(vd->offset);

        llvm::Constant* v = 0;

        if (ExpInitializer* ex = ini->isExpInitializer())
        {
            elem* e = ex->exp->toElem(gIR);
            v = llvm::cast<llvm::Constant>(e->val);
            delete e;
        }
        else if (StructInitializer* si = ini->isStructInitializer())
        {
            v = LLVM_DtoStructInitializer(si);
        }
        else if (ArrayInitializer* ai = ini->isArrayInitializer())
        {
            v = LLVM_DtoArrayInitializer(ai);
        }
        else if (ini->isVoidInitializer())
        {
            v = llvm::UndefValue::get(structtype->getElementType(idx));
        }
        else
        assert(v);

        inits[idx] = v;
    }

    // fill out nulls
    assert(si->ad->llvmInitZ);
    if (si->ad->llvmInitZ->isNullValue())
    {
        for (int i = 0; i < n; ++i)
        {
            if (inits[i] == 0)
            {
                inits[i] = llvm::Constant::getNullValue(structtype->getElementType(i));
            }
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            if (inits[i] == 0)
            {
                inits[i] = si->ad->llvmInitZ->getOperand(i);
            }
        }
    }

    return llvm::ConstantStruct::get(structtype, inits);
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

llvm::Value* LLVM_DtoNullDelegate(llvm::Value* v)
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

    llvm::Value* ret = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoDelegateCopy(llvm::Value* dst, llvm::Value* src)
{
    assert(dst->getType() == src->getType());
    assert(gIR);

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

        llvm::Value* dstlen = new llvm::GetElementPtrInst(dst,zero,zero,"tmp",gIR->scopebb());
        llvm::Value* srclen = llvm::ConstantInt::get(LLVM_DtoSize_t(), arrty->getNumElements(), false);
        new llvm::StoreInst(srclen, dstlen, gIR->scopebb());

        llvm::Value* dstptr = new llvm::GetElementPtrInst(dst,zero,one,"tmp",gIR->scopebb());
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
        args[0] = new llvm::GetElementPtrInst(l,zero,zero,"tmp",gIR->scopebb());
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
    const llvm::StructType* st = llvm::cast<llvm::StructType>(arr->getType()->getContainedType(0));
    //const llvm::PointerType* pt = llvm::cast<llvm::PointerType>(r->getType());
    
    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* arrdim = new llvm::GetElementPtrInst(arr,zero,zero,"tmp",gIR->scopebb());
    new llvm::StoreInst(dim, arrdim, gIR->scopebb());
    
    llvm::Value* arrptr = new llvm::GetElementPtrInst(arr,zero,one,"tmp",gIR->scopebb());
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

    llvm::ArrayType* arrty = LLVM_DtoStaticArrayType(t);
    return llvm::ConstantArray::get(arrty, inits);
}

//////////////////////////////////////////////////////////////////////////////////////////
void LLVM_DtoArrayCopy(elem* dst, elem* src)
{
    assert(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalValue::LinkageTypes LLVM_DtoLinkage(PROT prot, uint stc)
{
    switch(prot)
    {
    case PROTprivate:
        return llvm::GlobalValue::InternalLinkage;

    case PROTpublic:
    case PROTpackage:
    case PROTprotected:
    case PROTexport:
        return llvm::GlobalValue::ExternalLinkage;

    case PROTundefined:
    case PROTnone:
        assert(0 && "Unsupported linkage type");
    }
    return llvm::GlobalValue::ExternalLinkage;

/*      ExternalLinkage = 0, LinkOnceLinkage, WeakLinkage, AppendingLinkage,
  InternalLinkage, DLLImportLinkage, DLLExportLinkage, ExternalWeakLinkage,
  GhostLinkage */
}

//////////////////////////////////////////////////////////////////////////////////////////

unsigned LLVM_DtoCallingConv(LINK l)
{
    if (l == LINKc)
        return llvm::CallingConv::C;
    else if (l == LINKd || l == LINKdefault)
        return llvm::CallingConv::Fast;
    else if (l == LINKwindows)
        return llvm::CallingConv::X86_StdCall;
    else
        assert(0 && "Unsupported calling convention");
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoPointedType(llvm::Value* ptr, llvm::Value* val)
{
    const llvm::Type* ptrTy = ptr->getType()->getContainedType(0);
    const llvm::Type* valTy = val->getType();
    // ptr points to val's type
    if (ptrTy == valTy)
    {
        return val;
    }
    // ptr is integer pointer
    else if (ptrTy->isInteger())
    {
        // val is integer
        assert(valTy->isInteger());
        const llvm::IntegerType* pt = llvm::cast<const llvm::IntegerType>(ptrTy);
        const llvm::IntegerType* vt = llvm::cast<const llvm::IntegerType>(valTy);
        if (pt->getBitWidth() < vt->getBitWidth()) {
            return new llvm::TruncInst(val, pt, "tmp", gIR->scopebb());
        }
        else
        assert(0);
    }
    // something else unsupported
    else
    {
        Logger::cout() << *ptrTy << '|' << *valTy << '\n';
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* LLVM_DtoBoolean(llvm::Value* val)
{
    const llvm::Type* t = val->getType();
    if (t->isInteger())
    {
        if (t == llvm::Type::Int1Ty)
            return val;
        else {
            llvm::Value* zero = llvm::ConstantInt::get(t, 0, false);
            return new llvm::ICmpInst(llvm::ICmpInst::ICMP_NE, val, zero, "tmp", gIR->scopebb());
        }
    }
    else if (llvm::isa<llvm::PointerType>(t)) {
        const llvm::Type* st = LLVM_DtoSize_t();
        llvm::Value* ptrasint = new llvm::PtrToIntInst(val,st,"tmp",gIR->scopebb());
        llvm::Value* zero = llvm::ConstantInt::get(st, 0, false);
        return new llvm::ICmpInst(llvm::ICmpInst::ICMP_NE, ptrasint, zero, "tmp", gIR->scopebb());
    }
    else
    {
        Logger::cout() << *t << '\n';
    }
    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::Type* LLVM_DtoSize_t()
{
    if (global.params.is64bit)
    return llvm::Type::Int64Ty;
    else
    return llvm::Type::Int32Ty;
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoMain()
{
    // emit main function llvm style
    // int main(int argc, char**argv, char**env);

    assert(gIR != 0);
    IRState& ir = *gIR;

    assert(ir.emitMain && ir.mainFunc);

    // parameter types
    std::vector<const llvm::Type*> pvec;
    pvec.push_back((const llvm::Type*)llvm::Type::Int32Ty);
    const llvm::Type* chPtrType = (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
    pvec.push_back((const llvm::Type*)llvm::PointerType::get(chPtrType));
    pvec.push_back((const llvm::Type*)llvm::PointerType::get(chPtrType));
    const llvm::Type* rettype = (const llvm::Type*)llvm::Type::Int32Ty;

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, pvec, false);
    llvm::Function* func = new llvm::Function(functype,llvm::GlobalValue::ExternalLinkage,"main",ir.module);

    llvm::BasicBlock* bb = new llvm::BasicBlock("entry",func);

    // call static ctors
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(ir.module,"_d_run_module_ctors");
    new llvm::CallInst(fn,"",bb);

    // call user main function
    llvm::CallInst* call = new llvm::CallInst(ir.mainFunc,"ret",bb);
    call->setCallingConv(ir.mainFunc->getCallingConv());

    // call static dtors
    fn = LLVM_D_GetRuntimeFunction(ir.module,"_d_run_module_dtors");
    new llvm::CallInst(fn,"",bb);

    // return
    new llvm::ReturnInst(call,bb);

    /*
    // return value type
    const llvm::Type* rettype;
    Type* rt = f->next;
    if (rt) {
        rettype = LLVM_DtoType(rt);
    }
    else {
        assert(0);
    }

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, paramvec, false);
    */
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoCallClassDtors(TypeClass* tc, llvm::Value* instance)
{
    Array* arr = &tc->sym->dtors;
    for (size_t i=0; i<arr->dim; i++)
    {
        FuncDeclaration* fd = (FuncDeclaration*)arr->data[i];
        assert(fd->llvmValue);
        new llvm::CallInst(fd->llvmValue, instance, "", gIR->scopebb());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void LLVM_DtoInitClass(TypeClass* tc, llvm::Value* dst)
{
    assert(tc->llvmInit);
    assert(dst->getType() == tc->llvmInit->getType());
    assert(gIR);

    assert(tc->llvmType);
    uint64_t n = gTargetData->getTypeSize(tc->llvmType);
    llvm::Type* arrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
    llvm::Value* srcarr = new llvm::BitCastInst(tc->llvmInit,arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* LLVM_DtoInitializer(Type* type, Initializer* init)
{
    llvm::Constant* _init = 0;
    if (!init)
    {
        elem* e = type->defaultInit()->toElem(gIR);
        if (!e->inplace && !e->isNull()) {
            _init = llvm::cast<llvm::Constant>(e->getValue());
        }
        delete e;
    }
    else if (ExpInitializer* ex = init->isExpInitializer())
    {
        elem* e = ex->exp->toElem(gIR);
        if (!e->inplace && !e->isNull()) {
            _init = llvm::cast<llvm::Constant>(e->getValue());
        }
        delete e;
    }
    else if (StructInitializer* si = init->isStructInitializer())
    {
        _init = LLVM_DtoStructInitializer(si);
    }
    else if (ArrayInitializer* ai = init->isArrayInitializer())
    {
        _init = LLVM_DtoArrayInitializer(ai);
    }
    else if (init->isVoidInitializer())
    {
        const llvm::Type* ty = LLVM_DtoType(type);
        _init = llvm::Constant::getNullValue(ty);
    }
    else {
        Logger::println("unsupported initializer: %s", init->toChars());
    }
    return _init;
}
